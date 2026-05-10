//go:build js && wasm

// llm.go — SmolLM2-135M inference in pure Go, compiled to WebAssembly.
//
// Fetches SmolLM2-135M-Instruct-Q4_K_M.gguf (~105 MB) directly from
// HuggingFace. The GGUF file is self-contained: it embeds the BPE
// tokenizer vocab + merge rules alongside the quantized weights.
//
// This file implements from scratch:
//   GGUF binary parser · BPE tokenizer · Q4_K / Q6_K dequantization
//   RMSNorm · RoPE · grouped-query attention · SwiGLU MLP · top-k sampling
//
// Inspired by llm.c (Karpathy) and llm.cpp (ggerganov). No ML libraries.

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"syscall/js"
)

// ══════════════════════════════════════════════════════════════════════════════
// §1  GGUF PARSER
//
// Layout: magic(4) | version(4) | n_tensors(8) | n_kv(8) |
//         kv_pairs... | tensor_info... | <alignment pad> | tensor_data...
// ══════════════════════════════════════════════════════════════════════════════

const ggufMagic = 0x46554747 // little-endian 'GGUF'

// GGML tensor type IDs
const (
	typeF32  uint32 = 0
	typeF16  uint32 = 1
	typeQ4_0 uint32 = 2  // legacy 4-bit, QK=32
	typeQ5_0 uint32 = 6  // 5-bit, QK=32  ← used in Q4_K_M for some tensors
	typeQ8_0 uint32 = 8
	typeQ4K  uint32 = 12 // used by both Q4_K_S and Q4_K_M
	typeQ6K  uint32 = 14
)

// Block sizes for quantised types
const (
	QK_K  = 256 // elements per k-quant super-block
	szQ4K = 144 // bytes: fp16 d(2) + fp16 dmin(2) + scales(12) + qs(128)
	szQ6K = 210 // bytes: ql(128) + qh(64) + scales(16) + fp16 d(2)
	QK8   = 32  // elements per Q8_0 / Q4_0 / Q5_0 block
	szQ8  = 34  // bytes: fp16 d(2) + int8*32
	szQ4_0 = 18 // bytes: fp16 d(2) + qs[16]  (32 × 4-bit nibbles)
	szQ5_0 = 22 // bytes: fp16 d(2) + qh[4] + qs[16]  (32 × 5-bit)
)

// cur is a read-cursor over a byte slice.
type cur struct {
	d []byte
	p int
}

func (c *cur) u8() byte     { v := c.d[c.p]; c.p++; return v }
func (c *cur) u16() uint16  { v := binary.LittleEndian.Uint16(c.d[c.p:]); c.p += 2; return v }
func (c *cur) u32() uint32  { v := binary.LittleEndian.Uint32(c.d[c.p:]); c.p += 4; return v }
func (c *cur) u64() uint64  { v := binary.LittleEndian.Uint64(c.d[c.p:]); c.p += 8; return v }
func (c *cur) i32() int32   { return int32(c.u32()) }
func (c *cur) f32() float32 { return math.Float32frombits(c.u32()) }
func (c *cur) f64() float64 { return math.Float64frombits(c.u64()) }
func (c *cur) str() string {
	n := int(c.u64())
	s := string(c.d[c.p : c.p+n])
	c.p += n
	return s
}

// readVal reads one GGUF value given its type tag.
func (c *cur) readVal(tag uint32) any {
	switch tag {
	case 0:
		return c.u8()
	case 1:
		return int8(c.u8())
	case 2:
		return c.u16()
	case 3:
		return int16(c.u16())
	case 4:
		return c.u32()
	case 5:
		return c.i32()
	case 6:
		return c.f32()
	case 7:
		return c.u8() != 0 // bool
	case 8:
		return c.str()
	case 9: // array
		et := c.u32()
		n := c.u64()
		a := make([]any, n)
		for i := range a {
			a[i] = c.readVal(et)
		}
		return a
	case 10:
		return c.u64()
	case 11:
		return int64(c.u64())
	case 12:
		return c.f64()
	default:
		panic(fmt.Sprintf("unknown GGUF value type %d", tag))
	}
}

// ggufTensor describes one tensor's location and quantisation type.
type ggufTensor struct {
	name   string
	shape  []int  // innermost dim first (GGUF convention)
	typ    uint32
	offset uint64 // byte offset from start of data section
}

// ggufFile holds parsed GGUF metadata plus a reference to the raw bytes.
type ggufFile struct {
	kv      map[string]any
	tensors map[string]*ggufTensor
	raw     []byte
	dataOff int // byte index where tensor data begins
}

func parseGGUF(raw []byte) (*ggufFile, error) {
	c := &cur{d: raw}
	if c.u32() != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file")
	}
	ver := c.u32()
	if ver < 2 || ver > 3 {
		return nil, fmt.Errorf("GGUF version %d unsupported (need 2 or 3)", ver)
	}
	nT := c.u64()
	nKV := c.u64()

	kv := make(map[string]any, nKV)
	for i := uint64(0); i < nKV; i++ {
		key := c.str()
		kv[key] = c.readVal(c.u32())
	}

	tensors := make(map[string]*ggufTensor, nT)
	for i := uint64(0); i < nT; i++ {
		name := c.str()
		nd := c.u32()
		shape := make([]int, nd)
		for d := range shape {
			shape[d] = int(c.u64())
		}
		typ := c.u32()
		off := c.u64()
		tensors[name] = &ggufTensor{name, shape, typ, off}
	}

	align := uint64(32)
	if v, ok := kv["general.alignment"]; ok {
		switch a := v.(type) {
		case uint32:
			align = uint64(a)
		case uint64:
			align = a
		}
	}
	dataOff := int((uint64(c.p) + align - 1) / align * align)
	return &ggufFile{kv, tensors, raw, dataOff}, nil
}

// Typed KV accessors with defaults.
func (g *ggufFile) kvU32(k string, def uint32) uint32 {
	v, ok := g.kv[k]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case uint32:
		return x
	case uint64:
		return uint32(x)
	case int32:
		return uint32(x)
	}
	return def
}

func (g *ggufFile) kvF32(k string, def float32) float32 {
	v, ok := g.kv[k]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	}
	return def
}

func (g *ggufFile) kvArr(k string) []any {
	v, _ := g.kv[k]
	a, _ := v.([]any)
	return a
}

// tensorBytes returns the raw bytes slice for a named tensor.
func (g *ggufFile) tensorBytes(name string) ([]byte, *ggufTensor) {
	t, ok := g.tensors[name]
	if !ok {
		panic("tensor not found: " + name)
	}
	start := g.dataOff + int(t.offset)
	sz := tensorByteSize(t)
	return g.raw[start : start+sz], t
}

func tensorByteSize(t *ggufTensor) int {
	n := 1
	for _, d := range t.shape {
		n *= d
	}
	switch t.typ {
	case typeF32:
		return n * 4
	case typeF16:
		return n * 2
	case typeQ4_0:
		return (n / QK8) * szQ4_0
	case typeQ5_0:
		return (n / QK8) * szQ5_0
	case typeQ4K:
		return (n / QK_K) * szQ4K
	case typeQ6K:
		return (n / QK_K) * szQ6K
	case typeQ8_0:
		return (n / QK8) * szQ8
	}
	panic(fmt.Sprintf("unknown tensor type %d", t.typ))
}

// ══════════════════════════════════════════════════════════════════════════════
// §2  MODEL TYPES
// ══════════════════════════════════════════════════════════════════════════════

// Config holds Llama / SmolLM2 architectural hyperparameters.
type Config struct {
	nVocab, nCtx, nEmbd int
	nHeads, nKVHeads     int
	nLayers, nFF         int
	headDim              int
	ropeTheta            float32
	rmsEps               float32
}

// Weight is a slice of raw (quantised) bytes for one matrix, with metadata.
type Weight struct {
	data       []byte
	typ        uint32
	rows, cols int
}

func (w *Weight) rowBytes() int {
	return rowStride(w.typ, w.cols)
}

func rowStride(typ uint32, cols int) int {
	switch typ {
	case typeF32:
		return cols * 4
	case typeF16:
		return cols * 2
	case typeQ4_0:
		return (cols / QK8) * szQ4_0
	case typeQ5_0:
		return (cols / QK8) * szQ5_0
	case typeQ4K:
		return (cols / QK_K) * szQ4K
	case typeQ6K:
		return (cols / QK_K) * szQ6K
	case typeQ8_0:
		return (cols / QK8) * szQ8
	}
	panic(fmt.Sprintf("unsupported weight type %d for stride", typ))
}

// Layer holds references to all weight tensors in one transformer block.
type Layer struct {
	attnNorm []float32 // decoded F32 [nEmbd]
	ffnNorm  []float32
	attnQ    Weight // [nEmbd, nEmbd]
	attnK    Weight // [kvDim, nEmbd]
	attnV    Weight // [kvDim, nEmbd]
	attnO    Weight // [nEmbd, nEmbd]
	ffnGate  Weight // [nFF,   nEmbd]
	ffnUp    Weight // [nFF,   nEmbd]
	ffnDown  Weight // [nEmbd, nFF  ]
}

// ── global model state ────────────────────────────────────────────────────────

var (
	cfg        Config
	tokenEmbd  Weight    // [nVocab, nEmbd]
	lmHead     Weight    // [nVocab, nEmbd] (may alias tokenEmbd)
	outputNorm []float32 // [nEmbd]
	layers     []Layer

	// KV cache: [nLayers][maxCtx * kvDim]
	kvCacheK [][]float32
	kvCacheV [][]float32
	kvPos    int // number of tokens committed to the cache

	tok     *BPETokenizer
	isReady bool
	rng     = rand.New(rand.NewSource(42))
)

const maxCtx = 512

// ══════════════════════════════════════════════════════════════════════════════
// §3  QUANTISATION — fp16, Q4_K, Q6_K, Q8_0
// ══════════════════════════════════════════════════════════════════════════════

// fp16ToF32 converts IEEE 754 binary16 → binary32.
func fp16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 14)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | (e << 23) | ((mant &^ 0x400) << 13))
	}
	if exp == 0x1F {
		return math.Float32frombits(sign | 0x7F800000 | mant<<13)
	}
	return math.Float32frombits(sign | ((exp+112)<<23) | (mant << 13))
}

// scaleMinK4 extracts a 6-bit (scale, min) pair from the 12-byte Q4_K
// scales array.  j ∈ [0,7] is the sub-block index.
//
// Packing layout: bytes 0-3 hold the lower-6-bits of scale[0-3] and
// min[0-3] (in bytes 4-7); bytes 8-11 carry the upper-2-bits for
// indices 4-7, folded in by the else branch.
func scaleMinK4(j int, s []byte) (sc, m byte) {
	if j < 4 {
		sc, m = s[j]&63, s[j+4]&63
	} else {
		sc = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)
		m = (s[j+4] >> 4) | ((s[j] >> 6) << 4)
	}
	return
}

// dotQ4K computes dot(row, x) where row is Q4_K encoded.
// Block layout: fp16 d | fp16 dmin | scales[12] | qs[128]
// 8 sub-blocks of 32 elements each; lower nibble = first 16, upper nibble = next 16.
func dotQ4K(row []byte, x []float32) float32 {
	var acc float64
	for bi := 0; bi*QK_K < len(x); bi++ {
		b := row[bi*szQ4K:]
		d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
		dmin := float64(fp16ToF32(binary.LittleEndian.Uint16(b[2:])))
		sc := b[4:16]
		qs := b[16:]
		base := bi * QK_K
		for sb := 0; sb < 8; sb++ {
			scale, minv := scaleMinK4(sb, sc)
			db := d * float64(scale)
			mb := dmin * float64(minv)
			q := qs[sb*16:]
			xi := x[base+sb*32:]
			for l := 0; l < 16; l++ {
				acc += db*float64(q[l]&0xF)*float64(xi[l]) - mb*float64(xi[l])
				acc += db*float64(q[l]>>4)*float64(xi[l+16]) - mb*float64(xi[l+16])
			}
		}
	}
	return float32(acc)
}

// dotQ6K computes dot(row, x) where row is Q6_K encoded.
// Each 6-bit value = lower-4-bits from ql[] | upper-2-bits from qh[],
// centred by subtracting 32.  int8 scales, one per 16 elements.
func dotQ6K(row []byte, x []float32) float32 {
	var acc float64
	for bi := 0; bi*QK_K < len(x); bi++ {
		b := row[bi*szQ6K:]
		ql := b[0:128]
		qh := b[128:192]
		sc := b[192:208]
		d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[208:])))
		base := bi * QK_K
		for i := 0; i < QK_K; i++ {
			var lo byte
			if i&1 == 0 {
				lo = ql[i>>1] & 0xF
			} else {
				lo = ql[i>>1] >> 4
			}
			hi := (qh[i>>2] >> (2 * uint(i&3))) & 3
			q := (int(lo) | (int(hi) << 4)) - 32
			acc += d * float64(int8(sc[i>>4])) * float64(q) * float64(x[base+i])
		}
	}
	return float32(acc)
}

// dotQ8 computes dot(row, x) where row is Q8_0 encoded.
func dotQ8(row []byte, x []float32) float32 {
	var acc float64
	for bi := 0; bi*QK8 < len(x); bi++ {
		b := row[bi*szQ8:]
		d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
		base := bi * QK8
		for i := 0; i < QK8; i++ {
			acc += d * float64(int8(b[2+i])) * float64(x[base+i])
		}
	}
	return float32(acc)
}

func dotF32(row []byte, x []float32) float32 {
	var acc float64
	for i, xi := range x {
		acc += float64(math.Float32frombits(binary.LittleEndian.Uint32(row[i*4:]))) * float64(xi)
	}
	return float32(acc)
}

func dotF16(row []byte, x []float32) float32 {
	var acc float64
	for i, xi := range x {
		acc += float64(fp16ToF32(binary.LittleEndian.Uint16(row[i*2:]))) * float64(xi)
	}
	return float32(acc)
}

// dotQ4_0: QK=32, block = fp16 d(2) + qs[16] (4-bit nibbles, subtract 8)
func dotQ4_0(row []byte, x []float32) float32 {
	var acc float64
	for bi := 0; bi*QK8 < len(x); bi++ {
		b := row[bi*szQ4_0:]
		d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
		qs := b[2:]
		base := bi * QK8
		for i := 0; i < 16; i++ {
			acc += d * (float64(qs[i]&0xF) - 8) * float64(x[base+i])
			acc += d * (float64(qs[i]>>4) - 8) * float64(x[base+16+i])
		}
	}
	return float32(acc)
}

// dotQ5_0: QK=32, block = fp16 d(2) + qh[4] + qs[16]
// 5-bit value = lower 4 bits from qs nibble | upper 1 bit from qh, centred at 16.
func dotQ5_0(row []byte, x []float32) float32 {
	var acc float64
	for bi := 0; bi*QK8 < len(x); bi++ {
		b := row[bi*szQ5_0:]
		d  := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
		qh := b[2:6]
		qs := b[6:]
		base := bi * QK8
		for i := 0; i < 16; i++ {
			hi0 := (qh[i>>3] >> uint(i&7)) & 1
			hi1 := (qh[(i+16)>>3] >> uint((i+16)&7)) & 1
			q0 := (int(qs[i]&0xF) | (int(hi0) << 4)) - 16
			q1 := (int(qs[i]>>4)  | (int(hi1) << 4)) - 16
			acc += d * float64(q0) * float64(x[base+i])
			acc += d * float64(q1) * float64(x[base+16+i])
		}
	}
	return float32(acc)
}

// matVec computes y = W·x, returning y of length w.rows.
func matVec(w *Weight, x []float32) []float32 {
	y := make([]float32, w.rows)
	rbs := w.rowBytes()
	for r := 0; r < w.rows; r++ {
		row := w.data[r*rbs : (r+1)*rbs]
		switch w.typ {
		case typeF32:
			y[r] = dotF32(row, x)
		case typeF16:
			y[r] = dotF16(row, x)
		case typeQ4_0:
			y[r] = dotQ4_0(row, x)
		case typeQ5_0:
			y[r] = dotQ5_0(row, x)
		case typeQ4K:
			y[r] = dotQ4K(row, x)
		case typeQ6K:
			y[r] = dotQ6K(row, x)
		case typeQ8_0:
			y[r] = dotQ8(row, x)
		default:
			panic(fmt.Sprintf("unsupported weight type %d", w.typ))
		}
	}
	return y
}

// embedRow decodes one row of an embedding table to []float32.
func embedRow(w *Weight, idx int) []float32 {
	rbs := w.rowBytes()
	row := w.data[idx*rbs : (idx+1)*rbs]
	out := make([]float32, w.cols)
	switch w.typ {
	case typeF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(row[i*4:]))
		}
	case typeF16:
		for i := range out {
			out[i] = fp16ToF32(binary.LittleEndian.Uint16(row[i*2:]))
		}
	case typeQ4K:
		for bi := 0; bi*QK_K < len(out); bi++ {
			b := row[bi*szQ4K:]
			d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
			dmin := float64(fp16ToF32(binary.LittleEndian.Uint16(b[2:])))
			sc := b[4:16]
			qs := b[16:]
			base := bi * QK_K
			for sb := 0; sb < 8; sb++ {
				scale, minv := scaleMinK4(sb, sc)
				db := d * float64(scale)
				mb := dmin * float64(minv)
				q := qs[sb*16:]
				for l := 0; l < 16; l++ {
					out[base+sb*32+l] = float32(db*float64(q[l]&0xF) - mb)
					out[base+sb*32+16+l] = float32(db*float64(q[l]>>4) - mb)
				}
			}
		}
	case typeQ4_0:
		for bi := 0; bi*QK8 < len(out); bi++ {
			b := row[bi*szQ4_0:]
			d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
			qs := b[2:]
			base := bi * QK8
			for i := 0; i < 16; i++ {
				out[base+i]    = float32(d * (float64(qs[i]&0xF) - 8))
				out[base+16+i] = float32(d * (float64(qs[i]>>4)  - 8))
			}
		}
	case typeQ5_0:
		for bi := 0; bi*QK8 < len(out); bi++ {
			b := row[bi*szQ5_0:]
			d  := float64(fp16ToF32(binary.LittleEndian.Uint16(b[0:])))
			qh := b[2:6]
			qs := b[6:]
			base := bi * QK8
			for i := 0; i < 16; i++ {
				hi0 := (qh[i>>3] >> uint(i&7)) & 1
				hi1 := (qh[(i+16)>>3] >> uint((i+16)&7)) & 1
				out[base+i]    = float32(d * float64((int(qs[i]&0xF)|(int(hi0)<<4))-16))
				out[base+16+i] = float32(d * float64((int(qs[i]>>4) |(int(hi1)<<4))-16))
			}
		}
	case typeQ6K:
		for bi := 0; bi*QK_K < len(out); bi++ {
			b := row[bi*szQ6K:]
			ql := b[0:128]
			qh := b[128:192]
			sc := b[192:208]
			d := float64(fp16ToF32(binary.LittleEndian.Uint16(b[208:])))
			base := bi * QK_K
			for i := 0; i < QK_K; i++ {
				var lo byte
				if i&1 == 0 {
					lo = ql[i>>1] & 0xF
				} else {
					lo = ql[i>>1] >> 4
				}
				hi := (qh[i>>2] >> (2 * uint(i&3))) & 3
				q := (int(lo) | (int(hi) << 4)) - 32
				out[base+i] = float32(d * float64(int8(sc[i>>4])) * float64(q))
			}
		}
	}
	return out
}

// ══════════════════════════════════════════════════════════════════════════════
// §4  MATH PRIMITIVES
// ══════════════════════════════════════════════════════════════════════════════

// rmsNorm: y_i = (x_i / rms(x)) * w_i
func rmsNorm(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	s := float32(1.0 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * s * w[i]
	}
	return out
}

// softmax applies numerically stable in-place softmax.
func softmax(x []float32) {
	mx := x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - mx)))
		sum += float64(x[i])
	}
	for i := range x {
		x[i] = float32(float64(x[i]) / sum)
	}
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

func addIP(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// §5  ROPE — ROTARY POSITION EMBEDDING
// ══════════════════════════════════════════════════════════════════════════════

func applyRoPE(q, k []float32, pos int) {
	ropeVec(q, cfg.nHeads, cfg.headDim, pos)
	ropeVec(k, cfg.nKVHeads, cfg.headDim, pos)
}

func ropeVec(x []float32, nH, hd, pos int) {
	for h := 0; h < nH; h++ {
		head := x[h*hd:]
		for i := 0; i < hd/2; i++ {
			theta := float64(pos) * math.Pow(float64(cfg.ropeTheta), -float64(2*i)/float64(hd))
			cos, sin := float32(math.Cos(theta)), float32(math.Sin(theta))
			x0, x1 := head[2*i], head[2*i+1]
			head[2*i] = x0*cos - x1*sin
			head[2*i+1] = x0*sin + x1*cos
		}
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// §6  LLAMA FORWARD PASS  (with KV cache)
//
//   embedding → L × (RMSNorm → GQA-attention → residual →
//                    RMSNorm → SwiGLU-MLP    → residual) →
//   RMSNorm → lm_head → logits
// ══════════════════════════════════════════════════════════════════════════════

// forwardOne runs one token through the model, appends K/V to the cache,
// and returns logits [nVocab].
func forwardOne(tokenID int) []float32 {
	x := embedRow(&tokenEmbd, tokenID)
	pos := kvPos
	kvDim := cfg.nKVHeads * cfg.headDim
	scale := float32(1.0 / math.Sqrt(float64(cfg.headDim)))
	group := cfg.nHeads / cfg.nKVHeads

	for l := 0; l < cfg.nLayers; l++ {
		lw := &layers[l]

		// ── Attention ────────────────────────────────────────────────────
		xn := rmsNorm(x, lw.attnNorm, cfg.rmsEps)
		q := matVec(&lw.attnQ, xn) // [nHeads*headDim]
		k := matVec(&lw.attnK, xn) // [kvDim]
		v := matVec(&lw.attnV, xn) // [kvDim]
		applyRoPE(q, k, pos)

		copy(kvCacheK[l][pos*kvDim:], k)
		copy(kvCacheV[l][pos*kvDim:], v)

		seqLen := pos + 1
		attnOut := make([]float32, cfg.nEmbd)

		for h := 0; h < cfg.nHeads; h++ {
			kvH := h / group
			qH := q[h*cfg.headDim:]
			outH := attnOut[h*cfg.headDim:]

			scores := make([]float32, seqLen)
			for t := 0; t < seqLen; t++ {
				kH := kvCacheK[l][t*kvDim+kvH*cfg.headDim:]
				var dot float32
				for d := 0; d < cfg.headDim; d++ {
					dot += qH[d] * kH[d]
				}
				scores[t] = dot * scale
			}
			softmax(scores)

			for t := 0; t < seqLen; t++ {
				vH := kvCacheV[l][t*kvDim+kvH*cfg.headDim:]
				s := scores[t]
				for d := 0; d < cfg.headDim; d++ {
					outH[d] += s * vH[d]
				}
			}
		}
		addIP(x, matVec(&lw.attnO, attnOut))

		// ── SwiGLU MLP ──────────────────────────────────────────────────
		xn2 := rmsNorm(x, lw.ffnNorm, cfg.rmsEps)
		gate := matVec(&lw.ffnGate, xn2)
		up := matVec(&lw.ffnUp, xn2)
		for i := range gate {
			gate[i] = silu(gate[i]) * up[i]
		}
		addIP(x, matVec(&lw.ffnDown, gate))
	}

	logits := matVec(&lmHead, rmsNorm(x, outputNorm, cfg.rmsEps))
	kvPos++
	return logits
}

// ══════════════════════════════════════════════════════════════════════════════
// §7  BPE TOKENIZER  (GPT-2 style, initialised from GGUF metadata)
//
//   tokenizer.ggml.tokens — vocabulary strings
//   tokenizer.ggml.merges — "piece_a piece_b" merge rules in rank order
// ══════════════════════════════════════════════════════════════════════════════

type BPETokenizer struct {
	enc           map[string]int
	dec           []string
	merges        map[[2]string]int
	byteEnc       [256]rune
	byteDec       map[rune]byte
	bosID         int
	eosID         int
	pat           *regexp.Regexp
	specialTokens []string // matched literally, bypassing BPE splitting
}

// buildByteCodec creates GPT-2's byte↔unicode mapping.
// Every byte value (including control bytes) maps to a unique printable rune.
func buildByteCodec() ([256]rune, map[rune]byte) {
	var bs []int
	for b := 33; b <= 126; b++ {
		bs = append(bs, b)
	} // !"#...~
	for b := 161; b <= 172; b++ {
		bs = append(bs, b)
	} // ¡...¬
	for b := 174; b <= 255; b++ {
		bs = append(bs, b)
	} // ®...ÿ
	cs := make([]rune, len(bs))
	for i, b := range bs {
		cs[i] = rune(b)
	}
	n := 256
	for b := 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, rune(n))
			n++
		}
	}
	var enc [256]rune
	dec := make(map[rune]byte, 256)
	for i, b := range bs {
		enc[b] = cs[i]
		dec[cs[i]] = byte(b)
	}
	return enc, dec
}

func buildTokenizer(g *ggufFile) (*BPETokenizer, error) {
	tokArr := g.kvArr("tokenizer.ggml.tokens")
	if tokArr == nil {
		return nil, fmt.Errorf("no tokenizer.ggml.tokens in GGUF")
	}
	dec := make([]string, len(tokArr))
	enc := make(map[string]int, len(tokArr))
	for i, v := range tokArr {
		s := v.(string)
		dec[i] = s
		enc[s] = i
	}

	mergesArr := g.kvArr("tokenizer.ggml.merges")
	merges := make(map[[2]string]int, len(mergesArr))
	for rank, v := range mergesArr {
		s := v.(string)
		if sp := strings.IndexByte(s, ' '); sp >= 0 {
			merges[[2]string{s[:sp], s[sp+1:]}] = rank
		}
	}

	byteEnc, byteDec := buildByteCodec()
	bosID := int(g.kvU32("tokenizer.ggml.bos_token_id", 1))
	eosID := int(g.kvU32("tokenizer.ggml.eos_token_id", 2))

	// Collect special/control tokens that must be encoded atomically.
	// Strategy: token_type==3 (control) AND any <|...|>-shaped token.
	// We run both passes so neither is a silent fallback for the other.
	specialSet := make(map[string]bool)
	if ttArr := g.kvArr("tokenizer.ggml.token_type"); ttArr != nil {
		for i, v := range ttArr {
			var tt int32
			switch x := v.(type) {
			case int32:  tt = x
			case uint32: tt = int32(x)
			}
			if tt == 3 && i < len(dec) {
				specialSet[dec[i]] = true
			}
		}
	}
	// Always also include <|...|>-shaped tokens (ChatML markers for SmolLM2).
	for _, s := range dec {
		if len(s) > 4 && s[0] == '<' && s[1] == '|' && s[len(s)-1] == '>' && s[len(s)-2] == '|' {
			specialSet[s] = true
		}
	}
	var specialTokens []string
	for s := range specialSet {
		specialTokens = append(specialTokens, s)
	}
	// Sort longest first so e.g. <|im_start|> is matched before <|im
	for i := 0; i < len(specialTokens); i++ {
		for j := i + 1; j < len(specialTokens); j++ {
			if len(specialTokens[j]) > len(specialTokens[i]) {
				specialTokens[i], specialTokens[j] = specialTokens[j], specialTokens[i]
			}
		}
	}

	return &BPETokenizer{
		enc: enc, dec: dec, merges: merges,
		byteEnc: byteEnc, byteDec: byteDec,
		bosID: bosID, eosID: eosID,
		specialTokens: specialTokens,
		pat: regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`),
	}, nil
}

// encode converts text to token IDs using byte-level BPE.
// Special/control tokens (e.g. <|im_start|>) are matched atomically first,
// before the BPE regex splits the text into words.
func (t *BPETokenizer) encode(text string) []int {
	var ids []int
	// Split text into segments: special tokens and plain text chunks.
	remaining := text
	for remaining != "" {
		// Check for a special token at the current position.
		matched := false
		for _, st := range t.specialTokens {
			if strings.HasPrefix(remaining, st) {
				if id, ok := t.enc[st]; ok {
					ids = append(ids, id)
					remaining = remaining[len(st):]
					matched = true
					break
				}
			}
		}
		if matched {
			continue
		}
		// Find the next special token occurrence.
		nextAt := len(remaining)
		for _, st := range t.specialTokens {
			if idx := strings.Index(remaining, st); idx >= 0 && idx < nextAt {
				nextAt = idx
			}
		}
		// BPE-encode the plain chunk up to (but not including) the next special token.
		chunk := remaining[:nextAt]
		remaining = remaining[nextAt:]
		for _, word := range t.pat.FindAllString(chunk, -1) {
			chars := make([]string, len(word))
			for i, b := range []byte(word) {
				chars[i] = string(t.byteEnc[b])
			}
			for len(chars) > 1 {
				bestRank, bestI := math.MaxInt32, -1
				for i := 0; i < len(chars)-1; i++ {
					if r, ok := t.merges[[2]string{chars[i], chars[i+1]}]; ok && r < bestRank {
						bestRank, bestI = r, i
					}
				}
				if bestI < 0 {
					break
				}
				merged := chars[bestI] + chars[bestI+1]
				tmp := make([]string, 0, len(chars)-1)
				tmp = append(tmp, chars[:bestI]...)
				tmp = append(tmp, merged)
				tmp = append(tmp, chars[bestI+2:]...)
				chars = tmp
			}
			for _, piece := range chars {
				if id, ok := t.enc[piece]; ok {
					ids = append(ids, id)
				}
			}
		}
	}
	return ids
}

// decode converts token IDs back to a UTF-8 string.
// Normal BPE tokens use the byte-decoder. Special tokens (e.g. <|im_end|>)
// are emitted as their raw UTF-8 strings since they're already valid text.
func (t *BPETokenizer) decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.dec) {
			continue
		}
		piece := t.dec[id]
		// Try byte-level decode first (normal BPE tokens)
		var buf []byte
		allMapped := true
		for _, ch := range piece {
			if b, ok := t.byteDec[ch]; ok {
				buf = append(buf, b)
			} else {
				allMapped = false
				break
			}
		}
		if allMapped {
			sb.Write(buf)
		} else {
			// Special token: write as-is (already valid UTF-8)
			sb.WriteString(piece)
		}
	}
	return sb.String()
}

// tokenizeDebug returns a human-readable description of how text is tokenized.
func (t *BPETokenizer) tokenizeDebug(text string) string {
	ids := t.encode(text)
	var sb strings.Builder
	for i, id := range ids {
		if i > 0 {
			sb.WriteString("|")
		}
		if id >= 0 && id < len(t.dec) {
			sb.WriteString(fmt.Sprintf("[%d:%s]", id, t.dec[id]))
		} else {
			sb.WriteString(fmt.Sprintf("[%d:?]", id))
		}
	}
	return sb.String()
}

// ══════════════════════════════════════════════════════════════════════════════
// §8  SAMPLING
// ══════════════════════════════════════════════════════════════════════════════

// sampleTopK draws one token from the top-k logits scaled by temperature.
// temperature=0 → greedy argmax.
func sampleTopK(logits []float32, temperature float32, k int) int {
	if temperature == 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}
	tmp := make([]float32, len(logits))
	copy(tmp, logits)
	for i := range tmp {
		tmp[i] /= temperature
	}

	type iv struct {
		i int
		v float32
	}
	if k > len(tmp) {
		k = len(tmp)
	}
	top := make([]iv, 0, k)
	for i, v := range tmp {
		if len(top) < k {
			top = append(top, iv{i, v})
			for j := len(top) - 1; j > 0 && top[j].v > top[j-1].v; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		} else if v > top[k-1].v {
			top[k-1] = iv{i, v}
			for j := k - 1; j > 0 && top[j].v > top[j-1].v; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}
	probs := make([]float32, len(top))
	for i, p := range top {
		probs[i] = p.v
	}
	softmax(probs)
	r := rng.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r <= cum {
			return top[i].i
		}
	}
	return top[0].i
}

// ══════════════════════════════════════════════════════════════════════════════
// §9  MODEL LOADING
// ══════════════════════════════════════════════════════════════════════════════

func decodeF32Vec(g *ggufFile, name string) []float32 {
	data, t := g.tensorBytes(name)
	n := 1
	for _, d := range t.shape {
		n *= d
	}
	out := make([]float32, n)
	switch t.typ {
	case typeF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	case typeF16:
		for i := range out {
			out[i] = fp16ToF32(binary.LittleEndian.Uint16(data[i*2:]))
		}
	default:
		panic("norm weight must be F32 or F16")
	}
	return out
}

func makeWeight(g *ggufFile, name string, rows, cols int) Weight {
	data, t := g.tensorBytes(name)
	return Weight{data: data, typ: t.typ, rows: rows, cols: cols}
}

func loadModel(raw []byte) error {
	g, err := parseGGUF(raw)
	if err != nil {
		return fmt.Errorf("GGUF: %w", err)
	}

	cfg = Config{
		nVocab:    int(g.kvU32("llama.vocab_size", 49152)),
		nCtx:      int(g.kvU32("llama.context_length", 2048)),
		nEmbd:     int(g.kvU32("llama.embedding_length", 576)),
		nHeads:    int(g.kvU32("llama.attention.head_count", 9)),
		nKVHeads:  int(g.kvU32("llama.attention.head_count_kv", 3)),
		nLayers:   int(g.kvU32("llama.block_count", 30)),
		nFF:       int(g.kvU32("llama.feed_forward_length", 1536)),
		ropeTheta: g.kvF32("llama.rope.freq_base", 10000.0),
		rmsEps:    g.kvF32("llama.attention.layer_norm_rms_epsilon", 1e-5),
	}
	cfg.headDim = cfg.nEmbd / cfg.nHeads
	kvDim := cfg.nKVHeads * cfg.headDim

	tokenEmbd = makeWeight(g, "token_embd.weight", cfg.nVocab, cfg.nEmbd)
	outputNorm = decodeF32Vec(g, "output_norm.weight")
	if _, ok := g.tensors["output.weight"]; ok {
		lmHead = makeWeight(g, "output.weight", cfg.nVocab, cfg.nEmbd)
	} else {
		lmHead = tokenEmbd // tied weights
	}

	layers = make([]Layer, cfg.nLayers)
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		lw := &layers[l]
		lw.attnNorm = decodeF32Vec(g, p+"attn_norm.weight")
		lw.ffnNorm = decodeF32Vec(g, p+"ffn_norm.weight")
		lw.attnQ = makeWeight(g, p+"attn_q.weight", cfg.nEmbd, cfg.nEmbd)
		lw.attnK = makeWeight(g, p+"attn_k.weight", kvDim, cfg.nEmbd)
		lw.attnV = makeWeight(g, p+"attn_v.weight", kvDim, cfg.nEmbd)
		lw.attnO = makeWeight(g, p+"attn_output.weight", cfg.nEmbd, cfg.nEmbd)
		lw.ffnGate = makeWeight(g, p+"ffn_gate.weight", cfg.nFF, cfg.nEmbd)
		lw.ffnUp = makeWeight(g, p+"ffn_up.weight", cfg.nFF, cfg.nEmbd)
		lw.ffnDown = makeWeight(g, p+"ffn_down.weight", cfg.nEmbd, cfg.nFF)
	}

	kvCacheK = make([][]float32, cfg.nLayers)
	kvCacheV = make([][]float32, cfg.nLayers)
	for l := range kvCacheK {
		kvCacheK[l] = make([]float32, maxCtx*kvDim)
		kvCacheV[l] = make([]float32, maxCtx*kvDim)
	}
	kvPos = 0

	var terr error
	tok, terr = buildTokenizer(g)
	if terr != nil {
		return fmt.Errorf("tokenizer: %w", terr)
	}

	isReady = true
	return nil
}

// ══════════════════════════════════════════════════════════════════════════════
// §10  JS API
//
//   llm.load(arrayBuffer)           → "" ok | error string
//   llm.generate(prompt, n, temp)   → generated string
//   llm.ready()                     → bool
// ══════════════════════════════════════════════════════════════════════════════

var busy bool

func jsLoad(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return "missing ArrayBuffer argument"
	}
	n := args[0].Get("byteLength").Int()
	raw := make([]byte, n)
	js.CopyBytesToGo(raw, js.Global().Get("Uint8Array").New(args[0]))
	if err := loadModel(raw); err != nil {
		return err.Error()
	}
	return ""
}

func jsGenerate(_ js.Value, args []js.Value) any {
	if !isReady {
		return "model not loaded"
	}
	if busy {
		return "busy"
	}
	busy = true
	defer func() { busy = false }()

	prompt := "Hello"
	maxNew := 80
	temp := float32(0.8)
	if len(args) > 0 {
		prompt = args[0].String()
	}
	if len(args) > 1 {
		maxNew = args[1].Int()
	}
	if len(args) > 2 {
		temp = float32(args[2].Float())
	}

	// Format as ChatML for SmolLM2-Instruct
	full := "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n" +
		"<|im_start|>user\n" + prompt + "<|im_end|>\n" +
		"<|im_start|>assistant\n"

	kvPos = 0
	ids := tok.encode(full)
	if len(ids) == 0 {
		ids = []int{tok.bosID}
	}

	// Prefill the prompt
	var logits []float32
	for _, id := range ids {
		if kvPos >= maxCtx {
			break
		}
		logits = forwardOne(id)
	}

	// Autoregressive decode
	var out []int
	for i := 0; i < maxNew && kvPos < maxCtx; i++ {
		next := sampleTopK(logits, temp, 40)
		if next == tok.eosID {
			break
		}
		out = append(out, next)
		logits = forwardOne(next)
	}
	return tok.decode(out)
}

func jsReady(_ js.Value, _ []js.Value) any { return isReady }

func jsTokenize(_ js.Value, args []js.Value) any {
	if !isReady { return "not ready" }
	if len(args) < 1 { return "" }
	return tok.tokenizeDebug(args[0].String())
}

func main() {
	js.Global().Set("llm", js.ValueOf(map[string]any{
		"load":     js.FuncOf(jsLoad),
		"generate": js.FuncOf(jsGenerate),
		"ready":    js.FuncOf(jsReady),
		"tokenize": js.FuncOf(jsTokenize),
	}))
	js.Global().Get("document").Call("dispatchEvent",
		js.Global().Get("CustomEvent").New("wasmReady"),
	)
	select {} // keep the goroutine alive
}
