# DeckLink Rust (Blackmagic DeckLink SDK 14.4)

This project demonstrates using the Blackmagic DeckLink SDK from Rust via a C++ shim (FFI) to:
- List DeckLink devices
- Open a video capture stream and preview frames
  - CPU preview using `minifb`
  - GPU preview using `winit + wgpu`
  - Native macOS Cocoa Screen Preview that renders directly to an NSView for ultra-low latency

Note: The repo currently focuses on macOS and links against `DeckLinkAPI.framework`.

## Project Layout
- `Cargo.toml` — package, example binaries, and dependencies
- `build.rs` — builds the C++ shim and links `DeckLinkAPI.framework` (also adds `DeckLinkAPIDispatch.cpp`)
- `shim/shim.cpp` — C++ shim exposing a C ABI around DeckLink (device list, capture, screen preview)
- `include/` — DeckLink SDK headers plus `DeckLinkAPIDispatch.cpp` for macOS
- `src/lib.rs` — safe Rust wrapper(s) such as device listing
- Example binaries:
  - `src/bin/device_list.rs` — list devices
  - `src/bin/capture_preview.rs` — CPU preview (`minifb`)
  - `src/bin/capture_preview_wgpu.rs` — GPU preview (`wgpu`)
  - `src/bin/capture_preview_screen.rs` — Cocoa NSView screen preview

## System Requirements
- macOS with Blackmagic Desktop Video (driver) and DeckLink SDK installed
- `DeckLinkAPI.framework` available at `/Library/Frameworks/DeckLinkAPI.framework`
- Rust toolchain (`rustup`) and Xcode Command Line Tools

SDK setup options:
- The project includes `include/` with headers and `DeckLinkAPIDispatch.cpp`
- If your SDK is elsewhere, set the environment variable `DECKLINK_SDK_DIR` to the SDK root; `build.rs` searches multiple locations automatically

## Build
```bash
# Recommended: stable Rust
rustup default stable

# Build everything (including the C++ shim)
cargo build
# Or release build
cargo build --release
```

If `DeckLinkAPI.framework` is not under `/Library/Frameworks`, install it via Blackmagic Desktop Video or copy it there before building.

## Usage (example binaries)
The examples default to using the first device (index 0). To use another device, adjust the index in the example source for now.

- List devices
```bash
cargo run --bin devicelist
```

- CPU preview (convert to BGRA then display via `minifb`)
```bash
cargo run --bin capture_preview
```

- GPU preview via wgpu (Metal on macOS)
```bash
cargo run --bin capture_preview_wgpu
```

- Cocoa NSView screen preview (very low latency)
```bash
cargo run --bin capture_preview_screen
```
Press Esc to exit in preview modes.

## Technical Notes
- The C++ shim (`shim/shim.cpp`) exposes C ABI functions (`decklink_list_devices`, `decklink_capture_open`, `decklink_capture_get_frame`, `decklink_preview_attach_nsview`, etc.) for Rust to call
- Multiple pixel formats are converted to BGRA for CPU preview (e.g., UYVY, YUYV, v210 → BGRA)
- The Screen Preview path renders directly into an NSView via DeckLink, minimizing copies and latency
- `build.rs`:
  - Adds link search and links `DeckLinkAPI.framework`, `CoreFoundation`, `CoreVideo`
  - Compiles `shim/shim.cpp` and includes `DeckLinkAPIDispatch.cpp` when found in known locations (`include/`, `DECKLINK_SDK_DIR`, etc.)

## Troubleshooting
- Framework not found at build time
  - Ensure `/Library/Frameworks/DeckLinkAPI.framework` exists and Blackmagic Desktop Video is installed
- `DeckLinkAPIDispatch.cpp` missing
  - The repo includes it under `include/`. If removed, set `DECKLINK_SDK_DIR` correctly or restore the file
- No video shown at runtime
  - Check the input signal and cabling
  - Use `cargo run --bin devicelist` to confirm the device is detected
  - If multiple cards/ports are present, update the device index in the example

## Limitations
- macOS-focused (Metal/wgpu + DeckLinkAPI.framework)
- No CLI flags yet for selecting device/mode (edit the example source for now)
- Examples do not include audio/recording yet

## Credits and License
- DeckLink SDK and related files are copyright Blackmagic Design Pty. Ltd.; follow the SDK license
- This repository’s example code does not specify a separate license; add one if needed for your project

---

# DeckLink Rust (Blackmagic DeckLink SDK 14.4)

โปรเจ็กต์นี้เป็นตัวอย่างการใช้งาน Blackmagic DeckLink SDK จากภาษา Rust โดยเชื่อมผ่าน C++ shim (FFI) เพื่อ:
- ค้นหาอุปกรณ์ DeckLink (device list)
- เปิดสตรีมวิดีโอจากการ์ดและแสดงผลแบบพรีวิว
  - โหมดพรีวิวผ่าน CPU (`minifb`)
  - โหมดพรีวิวผ่าน GPU (`winit + wgpu`)
  - โหมดพรีวิวแบบแนบตรงกับ NSView ของ macOS (Cocoa Screen Preview) — ลดดีเลย์สูงสุด

หมายเหตุ: โค้ดในรีโปนี้โฟกัสที่ macOS (DeckLinkAPI.framework) เป็นหลักในตอนนี้

## โครงสร้างสำคัญของโปรเจ็กต์
- `Cargo.toml` — กำหนดแพ็กเกจ, ไบนารีตัวอย่าง และไลบรารีที่ใช้
- `build.rs` — คอมไพล์ C++ shim และแนบ DeckLinkAPI.framework รวมถึง `DeckLinkAPIDispatch.cpp`
- `shim/shim.cpp` — C++ shim ที่ห่อ DeckLink SDK เป็น C ABI (เช่น list devices, capture, screen preview)
- `include/` — เฮดเดอร์ DeckLink SDK และ `DeckLinkAPIDispatch.cpp` สำหรับ macOS
- `src/lib.rs` — ฟังก์ชัน Rust แบบปลอดภัยสำหรับดึงรายชื่ออุปกรณ์ เป็นต้น
- ตัวอย่างไบนารี:
  - `src/bin/device_list.rs` — แสดงรายชื่ออุปกรณ์
  - `src/bin/capture_preview.rs` — พรีวิวผ่าน CPU (`minifb`)
  - `src/bin/capture_preview_wgpu.rs` — พรีวิวผ่าน GPU (`wgpu`)
  - `src/bin/capture_preview_screen.rs` — พรีวิวแบบแนบ NSView (Cocoa)

## ข้อกำหนดระบบ
- macOS พร้อมติดตั้ง Blackmagic Desktop Video (ไดรเวอร์) และ DeckLink SDK
- มี `DeckLinkAPI.framework` ที่ `/Library/Frameworks/DeckLinkAPI.framework`
- ติดตั้ง Rust (ผ่าน `rustup`) และมี Xcode Command Line Tools

ทางเลือกการตั้งค่า SDK:
- โฟลเดอร์ `include/` ในโปรเจ็กต์มีเฮดเดอร์ + `DeckLinkAPIDispatch.cpp` มาให้แล้ว
- หากติดตั้ง SDK นอกตำแหน่งปกติ กำหนดตัวแปรแวดล้อม `DECKLINK_SDK_DIR` ให้ชี้ไปยังราก SDK (ตัว `build.rs` จะค้นหาเฮดเดอร์/ซอร์สจากหลายตำแหน่งให้อัตโนมัติ)

## การติดตั้งและคอมไพล์
```bash
# แนะนำให้ใช้ Rust stable
rustup default stable

# คอมไพล์ทั้งหมด (รวม C++ shim)
cargo build
# หรือแบบ release
cargo build --release
```

หาก `DeckLinkAPI.framework` ไม่อยู่ใต้ `/Library/Frameworks` ให้ติดตั้งจาก Blackmagic Desktop Video SDK หรือคัดลอกไปยังตำแหน่งดังกล่าวก่อนคอมไพล์

## การใช้งาน (ตัวอย่างไบนารี)
แอปตัวอย่างทุกตัวในตอนนี้เลือกใช้อุปกรณ์ตัวแรก (index 0) หากต้องการเลือกอุปกรณ์อื่น ให้แก้ไขโค้ดที่เกี่ยวข้องในไฟล์ตัวอย่างนั้นๆ

- แสดงรายชื่ออุปกรณ์
```bash
cargo run --bin devicelist
```

- พรีวิวแบบ CPU (คอนเวิร์ตเป็น BGRA แล้วแสดงผลด้วย `minifb`)
```bash
cargo run --bin capture_preview
```

- พรีวิวแบบ GPU ผ่าน wgpu (Metal บน macOS)
```bash
cargo run --bin capture_preview_wgpu
```

- พรีวิวแบบแนบ NSView (Cocoa Screen Preview; ดีเลย์ต่ำมาก)
```bash
cargo run --bin capture_preview_screen
```
กด Esc เพื่อปิดหน้าต่าง/ออกจากโปรแกรมในโหมดพรีวิว

## รายละเอียดด้านเทคนิคโดยย่อ
- C++ shim (`shim/shim.cpp`) ห่อ DeckLink SDK ให้เป็น C ABI ที่ Rust เรียกได้สะดวก (เช่น `decklink_list_devices`, `decklink_capture_open`, `decklink_capture_get_frame`, `decklink_preview_attach_nsview`)
- รองรับสัญญาณหลายแบบและคอนเวิร์ตเป็น BGRA สำหรับพรีวิว CPU (เช่น UYVY, YUYV, v210 → BGRA)
- โหมด Screen Preview ใช้ DeckLink API เรนเดอร์เข้าสู่ NSView โดยตรง (ลดการคัดลอกบัฟเฟอร์)
- `build.rs` จะ:
  - เพิ่มเส้นทางลิงก์ `DeckLinkAPI.framework`, `CoreFoundation`, `CoreVideo`
  - คอมไพล์ `shim/shim.cpp` และแนบ `DeckLinkAPIDispatch.cpp` ถ้าพบในตำแหน่งที่รู้จัก (`include/`, `DECKLINK_SDK_DIR`, ฯลฯ)

## การแก้ไขปัญหา (Troubleshooting)
- คอมไพล์ไม่เจอ DeckLink framework
  - ตรวจสอบว่ามี `/Library/Frameworks/DeckLinkAPI.framework` และติดตั้ง Blackmagic Desktop Video ถูกต้อง
- คอมไพล์ไม่พบ `DeckLinkAPIDispatch.cpp`
  - โปรเจ็กต์นี้มีไฟล์ใน `include/` แล้ว หากย้ายออก ให้กำหนดตัวแปร `DECKLINK_SDK_DIR` ให้อยู่ถูกที่ หรือคัดลอกไฟล์กลับมา
- รันแล้วไม่เห็นภาพ
  - ตรวจสอบสัญญาณอินพุตที่เข้าการ์ด DeckLink และสายสัญญาณ
  - ลองใช้ `cargo run --bin devicelist` เพื่อตรวจสอบว่าเห็นอุปกรณ์
  - ถ้าต่อหลายการ์ด/หลายพอร์ต ให้แก้ไข index อุปกรณ์ในซอร์สไบนารีที่ใช้งาน

## ข้อจำกัดปัจจุบัน
- โฟกัสที่ macOS เป็นหลัก (Metal/wgpu + DeckLinkAPI.framework)
- ยังไม่ได้ทำ CLI เลือกอุปกรณ์/โหมดรับสัญญาณ (แก้ชั่วคราวได้ในซอร์สของไบนารี)
- ยังไม่รองรับการบันทึกเสียง/ไฟล์ในตัวอย่าง

## เครดิตและสิทธิ์การใช้งาน
- DeckLink SDK และไฟล์ที่เกี่ยวข้องเป็นลิขสิทธิ์ของ Blackmagic Design Pty. Ltd. โปรดปฏิบัติตามเงื่อนไขใบอนุญาตของ SDK
- โค้ดตัวอย่างในรีโปนี้ไม่มีการระบุไลเซนส์แยก หากต้องการระบุไลเซนส์ โปรดเพิ่มตามความต้องการของโปรเจ็กต์