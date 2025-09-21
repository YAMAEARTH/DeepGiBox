# DeckLink Rust Toolkit

## Overview
This repository demonstrates how to drive Blackmagic Design DeckLink capture devices from Rust by bridging through a C++ shim. It shows how to enumerate hardware, open the video capture pipeline, and preview frames using OpenGL while keeping the capture and preview code paths reusable.

## Highlights
- Safe-ish Rust wrappers (`CaptureSession`, `PreviewGl`, and device listing helpers) around the DeckLink C ABI.
- Optional OpenGL preview that reuses the capture pipeline so the preview code never needs to duplicate capture setup.
- Single C++ shim (`shim/shim.cpp`) that translates between DeckLink COM interfaces and a plain C ABI callable from Rust.

## Directory Layout
- `Cargo.toml` – crate definition and binary targets.
- `build.rs` – compiles the C++ shim, wires DeckLink libraries, and adds include directories discovered from the SDK.
- `shim/` – C++ bridge code plus DeckLink SDK headers that are needed at build time.
- `src/lib.rs` – Rust entry point that exposes device listing and the reusable capture/preview modules.
- `src/capture.rs` – RAII wrapper over the capture API returning BGRA frame metadata.
- `src/preview_gl.rs` – wrapper for the global DeckLink OpenGL preview helper.
- `src/bin/` – sample binaries:
  - `devicelist.rs` – prints available DeckLink devices.
  - `capture.rs` – capture-only helper that logs metadata (including the raw pointer returned by the shim).
  - `preview_gl.rs` – launches a winit window and renders frames via the OpenGL preview helper.
  - `capture_preview_gl.rs` – legacy combined capture + preview sample kept for reference.

## Prerequisites
- A DeckLink device with Blackmagic Desktop Video drivers installed.
- DeckLink SDK 14.4 (or compatible) installed on the build machine.
- Rust toolchain (Rust 1.70+ recommended).
- Supported operating systems: macOS and Linux (Windows paths would require additional work in the shim).

## Building
```bash
rustup default stable
cargo build
# or
cargo build --release
```

## Environment Configuration
`build.rs` automatically searches common SDK locations. If the DeckLink SDK lives elsewhere, set `DECKLINK_SDK_DIR` to the SDK root before compiling:
```bash
export DECKLINK_SDK_DIR=/path/to/Blackmagic_DeckLink_SDK
cargo build
```

On macOS you may also need `DeckLinkAPI.framework` under `/Library/Frameworks`. On Linux ensure `DeckLinkAPI` is discoverable in the library search path.

## Running the Samples
The samples default to device index 0. Adjust the index in source if you need a different input.

- List devices:
  ```bash
  cargo run --bin devicelist
  ```
- Poll capture frames and log metadata once per second:
  ```bash
  cargo run --bin capture
  ```
- Preview frames in an OpenGL window (uses the reusable capture + preview wrappers):
  ```bash
  cargo run --bin preview_gl
  ```
- Run the legacy combined demo:
  ```bash
  cargo run --bin capture_preview_gl
  ```

Press `Esc` or close the window to exit preview binaries.

## Data Flow Notes
`CaptureSession::get_frame` returns a `RawCaptureFrame` whose `data` field is a pointer into a BGRA buffer owned by the C++ shim. The buffer is updated in place for each new frame, so copy the pixels if you need to keep them beyond the current call. The OpenGL preview path never touches CPU pixel data; it forwards the latest `IDeckLinkVideoFrame` to `IDeckLinkGLScreenPreviewHelper` for rendering.

## Troubleshooting
- **SDK not found at build time** – verify `DeckLinkAPI.framework` (macOS) or the Linux shared library is on the compiler search path, or set `DECKLINK_SDK_DIR`.
- **No video at runtime** – confirm the input signal and cabling, and ensure the correct device index is selected.
- **Pixel pointer confusion** – remember that `RawCaptureFrame::data` is not owned by Rust; copy it if you plan to access the pixels after the next capture callback.

## Credits
DeckLink SDK and related headers are copyright Blackmagic Design Pty. Ltd. This repository only contains sample code and does not ship DeckLink binaries.

---

# เด็คลิงก์ รัสต์ ทูลคิต

## ภาพรวม
รีโพนี้สาธิตการควบคุมอุปกรณ์จับสัญญาณ Blackmagic DeckLink ด้วยภาษา Rust โดยเชื่อมผ่าน C++ shim แสดงตัวอย่างการดึงรายชื่ออุปกรณ์ เปิดสตรีมวิดีโอ และพรีวิวเฟรมด้วย OpenGL โดยจัดโค้ดส่วน capture และ preview ให้ใช้ซ้ำได้

## จุดเด่น
- มีตัวห่อ Rust (`CaptureSession`, `PreviewGl` และฟังก์ชันรายการอุปกรณ์) เพื่อเรียก DeckLink ผ่าน C ABI ได้สะดวกขึ้น
- พรีวิวแบบ OpenGL ที่ใช้งานร่วมกับ capture เดิมได้ทันที ไม่ต้องตั้งค่าซ้ำซ้อน
- Shim C++ ไฟล์เดียว (`shim/shim.cpp`) ทำหน้าที่แปลง DeckLink COM เป็น C ABI ให้ Rust เรียกใช้

## โครงสร้างโฟลเดอร์
- `Cargo.toml` – นิยามครตและไบนารีตัวอย่าง
- `build.rs` – คอมไพล์ C++ shim และตั้งค่าไลบรารี DeckLink
- `shim/` – โค้ด bridge ภาษา C++ และส่วนหัว DeckLink ที่ต้องใช้ตอนคอมไพล์
- `src/lib.rs` – จุดรวมฝั่ง Rust ที่เผยแพร่โมดูล capture/preview และฟังก์ชันรายการอุปกรณ์
- `src/capture.rs` – ตัวห่อ capture แบบ RAII ที่คืนค่าเมตาดาตาเฟรม BGRA
- `src/preview_gl.rs` – ตัวห่อ DeckLink OpenGL preview helper
- `src/bin/` – ไบนารีตัวอย่าง:
  - `devicelist.rs` – แสดงรายชื่อ DeckLink ที่พบ
  - `capture.rs` – โหมด capture-only แสดง pointer และข้อมูลเฟรมทุกวินาที
  - `preview_gl.rs` – เปิดหน้าต่าง winit และเรนเดอร์เฟรมผ่าน OpenGL preview
  - `capture_preview_gl.rs` – ตัวอย่างแบบรวม capture + preview เดิมเพื่ออ้างอิง

## ความต้องการระบบ
- มีการ์ด DeckLink และติดตั้งไดรเวอร์ Blackmagic Desktop Video แล้ว
- ติดตั้ง DeckLink SDK 14.4 (หรือเวอร์ชันที่รองรับ) บนเครื่องพัฒนา
- ใช้ Rust toolchain (แนะนำรุ่น stable)
- รองรับ macOS และ Linux (ถ้าใช้ Windows ต้องปรับ shim เพิ่ม)

## การคอมไพล์
```bash
rustup default stable
cargo build
# หรือ
cargo build --release
```

## การตั้งค่าสภาพแวดล้อม
`build.rs` จะค้นหาไดเร็กทอรี SDK อัตโนมัติ แต่ถ้าอยู่ตำแหน่งอื่นให้ตั้ง `DECKLINK_SDK_DIR` ชี้ไปยังโฟลเดอร์ SDK ก่อนคอมไพล์:
```bash
export DECKLINK_SDK_DIR=/path/to/Blackmagic_DeckLink_SDK
cargo build
```

บน macOS ต้องมี `DeckLinkAPI.framework` ที่ `/Library/Frameworks` ส่วน Linux ต้องให้ลิงก์ไลบรารี `DeckLinkAPI` เจอใน path ของระบบ

## การรันตัวอย่าง
ค่าปริยายจะใช้ device index 0 หากต้องการช่องสัญญาณอื่นให้แก้ในซอร์สก่อนรัน

- แสดงรายชื่ออุปกรณ์:
  ```bash
  cargo run --bin devicelist
  ```
- จับภาพและพิมพ์ข้อมูลเฟรมทุกหนึ่งวินาที:
  ```bash
  cargo run --bin capture
  ```
- พรีวิวผ่านหน้าต่าง OpenGL:
  ```bash
  cargo run --bin preview_gl
  ```
- รันตัวอย่างแบบรวมดั้งเดิม:
  ```bash
  cargo run --bin capture_preview_gl
  ```

กด `Esc` หรือปิดหน้าต่างเพื่อออกจากโหมดพรีวิว

## หมายเหตุเรื่องข้อมูล
`CaptureSession::get_frame` จะคืน `RawCaptureFrame` ซึ่ง `data` เป็น pointer ไปยังบัฟเฟอร์ BGRA ของ shim (รีเฟรชทุกเฟรม) หากต้องเก็บไว้ใช้นานๆ ต้องคัดลอกข้อมูลออกเอง ส่วนเส้นทางพรีวิว OpenGL จะส่ง `IDeckLinkVideoFrame` ต่อไปให้ DeckLink ช่วยวาดโดยตรง

## การแก้ปัญหาเบื้องต้น
- **คอมไพล์ไม่เจอ SDK** – ตรวจสอบตำแหน่ง `DeckLinkAPI.framework` (macOS) หรือไลบรารี (Linux) และตั้ง `DECKLINK_SDK_DIR` ให้ถูก
- **ไม่เห็นภาพเมื่อรัน** – ตรวจสอบสัญญาณ อินพุต และดัชนีอุปกรณ์ที่เลือกใช้
- **สับสนเรื่อง pointer** – จำไว้ว่าข้อมูลใน `RawCaptureFrame::data` ไม่ได้เป็นของ Rust ต้องก๊อปปี้ก่อนถ้าจะใช้นานๆ

## เครดิต
DeckLink SDK และไฟล์ที่เกี่ยวข้องเป็นลิขสิทธิ์ของ Blackmagic Design Pty. Ltd. รีโพนี้มีเพียงโค้ดตัวอย่างและไม่ได้แจกจ่ายไบนารี DeckLink
