
# Postprocessing Guideline (DeepGIBox)
**Stage:** `postprocess` (model outputs → `DetectionsPacket`)  
**Scope:** ไม่ขึ้นกับ TensorRT/ORT (ใช้ได้กับ inference backend ใด ๆ)  
**Version:** v1

เอกสารนี้กำหนดสเปกเชิงปฏิบัติสำหรับโมดูล `postprocess` ของ DeepGIBox ให้ทีมสามารถ implement ได้ตรงและทดสอบได้ง่าย โดยยึด `common_io` เป็นสัญญากลาง

---

## 1) เป้าหมาย
- แปลงผลลัพธ์ดิบจากโมเดล (tensor/logits) → ข้อมูลตรรกะพร้อมใช้งาน: รายการวัตถุ (bbox/score/class/track)
- รองรับหลายสถาปัตย์โมเดล: **YOLO‑style**, **DETR‑style**, **Segmentation**, **Keypoints**
- รองรับทั้ง **1080p60** และ **4Kp30** ด้วย latency ต่ำ (CPU หรือ GPU)
- คืนค่าเป็น `DetectionsPacket` สำหรับ `overlay_plan` และขั้นต่อไป

---

## 2) สัญญา I/O (Contracts)

### Input: `RawDetectionsPacket`
- แหล่งกำเนิดเฟรม: `from: FrameMeta { width, height, frame_idx, pts_ns, ... }`
- ประกอบด้วย **ตัวชี้เทนเซอร์** (CPU/GPU), รูปร่าง (shape), dtype (Fp16/Fp32), และชื่อเอาต์พุต
- ตัวอย่าง layout ที่พบบ่อย
  - **YOLO-style:** `outputs[0]: [1, N, (cx, cy, w, h, obj, class_0..class_C-1)]`
  - **DETR-style:** `boxes: [1, N, 4]` (normalized cx,cy,w,h); `logits: [1, N, C]`
  - **Segmentation:** `mask_logits: [1, C, Hm, Wm]` (+อาจมี det/boxes)
  - **Keypoints:** `kpt: [1, K, 3]` (x,y,score) หรือ heatmap

> แนะนำมี `TensorView { ptr, len, shape: Vec<i64>, dtype, device }` เพื่ออ่านค่าอย่างปลอดภัย (ดึง D2H/D2D เท่าที่จำเป็น)

### Output: `DetectionsPacket`
```rust
pub struct DetectionsPacket {
  pub from: FrameMeta,
  pub items: Vec<Detection>,
}
pub struct Detection {
  pub bbox: BBox { x: f32, y: f32, w: f32, h: f32 }, // พิกัดพิกเซลบนเฟรมจริง
  pub score: f32,
  pub class_id: i32,
  pub track_id: Option<i32>, // ถ้ามี tracking
  // (ออปชัน) pub extra: SmallVec<[f32; 8]>, // เผื่อ keypoints/attrs
}
```

---

## 3) ขั้นตอนหลักของ Postprocess

### 3.1 Decode (ขึ้นกับโมเดล)
- **YOLO-style**
  - raw: `(cx,cy,w,h,obj,cls0..clsC-1)` (sigmoid บางช่อง)
  - `score = obj * max(softmax_or_sigmoid(cls))`
  - `class_id = argmax(cls)`
  - แปลงเป็น **xywh (มุมบนซ้าย)**: `x = cx - w/2`, `y = cy - h/2`
- **DETR-style**
  - `boxes` normalized ∈ [0,1], `(cx,cy,w,h)`
  - `scores = softmax(logits)`; เลือก class+score สูงสุด (skip background idx)
- **Segmentation**
  - ถ้าเป็น instance seg มี `boxes` อยู่แล้ว → เก็บ `mask_logits`/`rle` ติด `Detection.extra` หรือสร้าง `Poly`
  - ถ้าเป็น semantic seg: สร้าง `Poly`/bitmap ในขั้นตอนถัดไป หรือแปลงเป็น overlays โดยตรง (ข้าม `DetectionsPacket` ได้ถ้า pipeline อนุญาต)
- **Keypoints**
  - ถ้าเป็น heatmap: argmax per joint → (x,y,score)
  - ติด keypoints ใน `Detection.extra` หรือสร้าง packet แยก (ตามดีไซน์ของคุณ)

> ทำ **Top‑K prefilter** ก่อน NMS เมื่อ N ใหญ่มาก (เช่น YOLO anchors 25k): partial sort โดยคะแนนเร็ว ๆ แล้วส่ง K~1–2k เข้า NMS

### 3.2 Remap พิกัดกลับสู่เฟรมจริง
รองรับ 2 โหมด preprocess:
- **Letterbox (คงสัดส่วน + เติมขอบ):**
  ```text
  x0 = (cx - w/2) * scale_x - pad_x
  y0 = (cy - h/2) * scale_y - pad_y
  w' = w * scale_x
  h' = h * scale_y
  ```
- **Warp resize (บิดสัดส่วน):** ใช้สัดส่วน `sx = W_frame / W_in`, `sy = H_frame / H_in` ตรง ๆ

สุดท้าย **clamp** กล่องให้อยู่ใน `[0, W_frame] × [0, H_frame]`

### 3.3 Thresholding
- `score >= cfg.score_thresh` (เช่น 0.25)
- (ออปชัน) กรองตามขนาดกล่องขั้นต่ำและอัตราส่วนกว้างยาว

### 3.4 NMS (Non‑Maximum Suppression)
- **Class-wise Greedy NMS** (ค่าเริ่มต้น):
  - แยกทีละ class → sort ตาม `score` → เก็บใบแรก แล้วลบใบที่มี IoU > `iou_thresh`
- ทางเลือก:
  - **Soft‑NMS:** ลด `score` ตาม IoU แทนการลบทิ้ง
  - **DIoU/CIoU‑NMS:** ใช้เมตริกที่คำนึงถึงระยะห่าง/ความครอบคลุม
- พารามิเตอร์ทั่วไป:
  - `cfg.nms.iou_thresh` (เช่น 0.45)
  - `cfg.nms.max_per_class`, `cfg.nms.max_total`
- สำหรับ N ใหญ่/4K: พิจารณา **GPU NMS** (CUDA) แล้วค่อย D2H ผลสุดท้าย

### 3.5 (ออปชัน) Tracking
- เพิ่ม `track_id` ด้วยตัวติดตามแบบเบา (IoU + Kalman) หรือ BYTETrack mini
- คอนฟิก: `tracking.enable`, `iou_match`, `max_age`

---

## 4) สูตรคอนฟิก (TOML)
```toml
[postprocess]
type          = "yolo"        # หรือ "detr" | "seg" | "keypoints"
score_thresh  = 0.25
max_dets      = 300

[postprocess.nms]
type          = "classwise"   # "soft" | "diou" | "ciou"
iou_thresh    = 0.45
max_per_class = 100
max_total     = 300

[tracking]
enable        = false
iou_match     = 0.5
max_age       = 30
```

---

## 5) โครง API (Rust)

```rust
pub struct Postprocess {
  cfg: config::PostCfg,
  // working buffers ที่ reuse ได้ เพื่อเลี่ยง alloc ใน hot path
}

impl Postprocess {
  pub fn new(cfg: config::PostCfg) -> anyhow::Result<Self> { /* init buffers */ }

  fn decode(&mut self, r: &RawDetectionsPacket) -> Vec<Candidate> { /* per-model */ }
  fn remap_to_frame(&self, cand: &mut [Candidate], fm: &FrameMeta) { /* letterbox/warp */ }
  fn threshold(&self, cand: &mut Vec<Candidate>, thr: f32) { /* retain */ }
  fn nms(&mut self, cand: &mut [Candidate], nms: &config::NmsCfg) -> Vec<Detection> { /* see §6 */ }
  fn track(&mut self, dets: Vec<Detection>) -> Vec<Detection> { /* optional */ }
}

impl common_io::Stage<RawDetectionsPacket, DetectionsPacket> for Postprocess {
  fn process(&mut self, r: RawDetectionsPacket) -> DetectionsPacket {
    let t0 = telemetry::now_ns();

    let mut cand = self.decode(&r);
    self.remap_to_frame(&mut cand, &r.from);
    self.threshold(&mut cand, self.cfg.score_thresh);
    let mut dets = self.nms(&mut cand, &self.cfg.nms);
    if self.cfg.tracking.enable { dets = self.track(dets); }

    telemetry::record_ms("postprocess", t0);
    DetectionsPacket { from: r.from, items: dets }
  }
}
```

โครงสร้างช่วยภายใน:
```rust
struct Candidate {
  cx:f32, cy:f32, w:f32, h:f32, score:f32, class_id:i32
}
#[inline] fn iou(a:&BBox, b:&BBox)->f32 { /* ดู §6 */ }
```

---

## 6) อัลกอริทึม Greedy NMS (ตัวอย่างโค้ดสั้น)

```rust
fn nms_classwise(mut dets: Vec<Detection>, iou_thr: f32, max_keep: usize) -> Vec<Detection> {
    dets.sort_by(|a,b| b.score.total_cmp(&a.score)); // คะแนนมากก่อน
    let mut keep = Vec::with_capacity(dets.len());
    'outer: for i in 0..dets.len() {
        if keep.len() == max_keep { break; }
        let di = &dets[i];
        for dj in &keep {
            if di.class_id == dj.class_id && iou(&di.bbox, &dj.bbox) > iou_thr {
                continue 'outer; // ทิ้ง di
            }
        }
        keep.push(dets[i].clone());
    }
    keep
}

#[inline]
fn iou(a:&BBox, b:&BBox)->f32 {
    let (ax1, ay1, ax2, ay2) = (a.x, a.y, a.x + a.w, a.y + a.h);
    let (bx1, by1, bx2, by2) = (b.x, b.y, b.x + b.w, b.y + b.h);
    let ix1 = ax1.max(bx1); let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2); let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0); let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = a.w*a.h + b.w*b.h - inter + 1e-6;
    inter / union
}
```

> สำหรับ **Soft‑NMS** ให้ลด `score` ของกล่องที่ IoU สูงด้วย Gaussian/Linear decay แทนการทิ้ง

---

## 7) ประสิทธิภาพ & หน่วยความจำ

- ใช้ **preallocated buffers** / `SmallVec` เพื่อลด alloc
- ทำ **Top‑K prefilter** ก่อน NMS เมื่อ N ใหญ่
- ลด transcendentals (ใช้ fast sigmoid/exp เมื่อเหมาะ)
- สำหรับ 4K/N ใหญ่: พิจารณา **GPU decode/NMS** → ดึงผลสุดท้ายเท่านั้นลง CPU
- เป้าหมายโดยประมาณ (ขึ้นกับฮาร์ดแวร์/โมเดล):
  - CPU NMS ~300–1000 boxes: **< 1–2 ms**
  - GPU NMS 10k+ proposals: **แนะนำ**

---

## 8) Telemetry ที่ควรมี
- `postprocess` (รวม)
- (ออปชัน) แยก subspan: `post.decode`, `post.nms`, `post.track`

---

## 9) Unit Tests (แนะนำ)
- **Decode correctness:** mock tensor ที่รู้ผล → ตรวจ bbox/score/class
- **NMS invariants:** IoU=1 (ซ้อนทับ) เหลือใบเดียว / ปรับ `iou_thresh` แล้วผลสอดคล้อง
- **Coord mapping:** เคส letterbox/warp → พิกัดตรงพิกเซลบนเฟรมจริง
- **Performance smoke:** N=1k → NMS ภายในงบเวลาที่ตั้ง (เช่น <2 ms)

ตัวอย่างสั้น:
```rust
#[test]
fn nms_behaves() {
  let cfg = default_cfg();
  let mut p = Postprocess::new(cfg).unwrap();
  let r = testsupport::yolo_mock_rawdet( /* N=200 มีซ้ำ */ );
  let out = p.process(r);
  assert!(out.items.len() <= p.cfg.nms.max_total);
  // ไม่มีสองกล่อง class เดียวกันที่ IoU > iou_thresh
}
```

---

## 10) ข้อผิดพลาดที่พบบ่อย
- ลืม remap พิกัดกลับสู่เฟรมจริง → กล่องวางผิดตำแหน่ง
- ทำ NMS รวมทุกคลาส → วัตถุต่างคลาสถูกลบกันเอง
- ไม่ clamp bbox → ค่าลบ/เกินขอบภาพ ทำให้วาด overlay พัง
- จัดการ dtype ผิด (Fp16→Fp32) → คะแนนเพี้ยน
- alloc ใหม่ทุกรอบเฟรม → jitter สูง

---

## 11) Checklist ก่อนใช้งานจริง
- [ ] รองรับรูปแบบโมเดลที่ใช้ (YOLO/DETR/Seg/Keypoints)
- [ ] Remap พิกัดถูก (รองรับ letterbox/warp) + clamp
- [ ] Threshold + NMS ถูกต้อง (class-wise เป็น default)
- [ ] (ถ้าเปิด) Tracking เติม `track_id`
- [ ] Telemetry ครบและไม่มี alloc ใน hot path
- [ ] Unit tests ผ่านทั้ง correctness และงบเวลาเบื้องต้น

---

**End of spec.**
