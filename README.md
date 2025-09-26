# DeckLink Rust (Blackmagic DeckLink SDK 14.4)

This project demonstrates using the Blackmagic DeckLink SDK from Rust via a C++ shim (FFI) to:
- List DeckLink devices
- Open a video capture stream and preview frames
  - CPU preview using `minifb`
  - OpenGL preview using the DeckLink GL screen preview helper
- Inspect 8-bit YUV frame buffers without conversion

Note: The repo currently targets Linux and links against the DeckLink SDK shared library (`libDeckLinkAPI.so`).

## Project Layout
- `Cargo.toml` — package, example binaries, and dependencies
- `build.rs` — builds the C++ shim and links `libDeckLinkAPI.so` (also adds `DeckLinkAPIDispatch.cpp`)
- `shim/shim.cpp` — C++ shim exposing a C ABI around DeckLink (device list, capture, OpenGL preview)
- `include/` — DeckLink SDK headers plus `DeckLinkAPIDispatch.cpp`
- `src/lib.rs` — safe Rust wrapper(s) such as device listing
- Example binaries:
  - `src/bin/device_list.rs` — list devices
  - `src/bin/capture_preview_gl.rs` — OpenGL preview via DeckLinkGLScreenPreviewHelper
  - `src/bin/capture_test.rs` — minimal capture loop for experiments
  - `src/bin/capture_yuv_dump.rs` — inspect DeckLink 8-bit YUV frames

## System Requirements
- Linux with Blackmagic Desktop Video (driver) and DeckLink SDK installed
- `libDeckLinkAPI.so` available on the loader search path (e.g., `/usr/lib`, `/usr/local/lib`, or via `LD_LIBRARY_PATH`)
- Rust toolchain (`rustup`) and standard build tools (e.g., `build-essential`)

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

Ensure `libDeckLinkAPI.so` can be found at runtime; adjust `LD_LIBRARY_PATH` if the library lives outside standard locations.

## Usage (example binaries)
The examples default to using the first device (index 0). To use another device, adjust the index in the example source for now.

- List devices
```bash
cargo run --bin devicelist
```

- OpenGL preview (DeckLink GL helper)
```bash
cargo run --bin capture_preview_gl
```

- Dump 8-bit YUV frames to console
```bash
cargo run --bin capture_yuv_dump
```

- Minimal capture loop for testing
```bash
cargo run --bin capture_test
```
Press Esc to exit in preview modes.

## Technical Notes
- The C++ shim (`shim/shim.cpp`) exposes C ABI functions (`decklink_list_devices`, `decklink_capture_open`, `decklink_capture_get_frame`, `decklink_capture_close`, and DeckLink GL preview helpers) for Rust to call
- DeckLink input is configured to deliver raw 8-bit YUV (UYVY); the shim copies each row verbatim into a shared buffer for Rust consumers
- `build.rs`:
  - Adds native link search paths and links against `libDeckLinkAPI`
  - Compiles `shim/shim.cpp` and includes `DeckLinkAPIDispatch.cpp` when found in common locations (`include/`, `DECKLINK_SDK_DIR`, etc.)

## Troubleshooting
- Shared library not found at build or run time
  - Ensure `libDeckLinkAPI.so` is installed (part of Desktop Video) and visible to both the linker and runtime loader (update `LD_LIBRARY_PATH` or copy it into `/usr/lib`/`/usr/local/lib`)
- `DeckLinkAPIDispatch.cpp` missing
  - The repo includes it under `include/`. If removed, set `DECKLINK_SDK_DIR` correctly or restore the file
- No video shown at runtime
  - Check the input signal and cabling
  - Use `cargo run --bin devicelist` to confirm the device is detected
  - If multiple cards/ports are present, update the device index in the example

## Limitations
- Linux-only for now (tested with Desktop Video 14.x)
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
  - โหมดพรีวิวผ่าน OpenGL ด้วย DeckLink GL screen preview helper
- ดึงเฟรม YUV 8-bit ไปจัดเก็บ/ประมวลผลภายหลัง

หมายเหตุ: ตอนนี้รีโปโฟกัสการทำงานบน Linux โดยเชื่อมกับไลบรารี `libDeckLinkAPI.so`

## โครงสร้างสำคัญของโปรเจ็กต์
- `Cargo.toml` — กำหนดแพ็กเกจ, ไบนารีตัวอย่าง และไลบรารีที่ใช้
- `build.rs` — คอมไพล์ C++ shim และลิงก์กับ `libDeckLinkAPI.so` (พร้อมเพิ่ม `DeckLinkAPIDispatch.cpp` หากพบ)
- `shim/shim.cpp` — C++ shim ที่ห่อ DeckLink SDK เป็น C ABI (list devices, capture, OpenGL preview)
- `include/` — เฮดเดอร์ DeckLink SDK และ `DeckLinkAPIDispatch.cpp`
- `src/lib.rs` — ฟังก์ชัน Rust แบบปลอดภัยสำหรับดึงรายชื่ออุปกรณ์ เป็นต้น
- ตัวอย่างไบนารี:
  - `src/bin/device_list.rs` — แสดงรายชื่ออุปกรณ์
  - `src/bin/capture_preview_gl.rs` — พรีวิวผ่าน OpenGL (DeckLinkGLScreenPreviewHelper)
  - `src/bin/capture_test.rs` — ลูปจับภาพแบบเรียบง่ายสำหรับทดลอง
  - `src/bin/capture_yuv_dump.rs` — ตรวจสอบเฟรม YUV 8-bit

## ข้อกำหนดระบบ
- Linux ที่ติดตั้ง Blackmagic Desktop Video (ไดรเวอร์) และ DeckLink SDK
- มี `libDeckLinkAPI.so` อยู่ในเส้นทางที่ตัวลิงเกอร์/รันไทม์มองเห็น (เช่น `/usr/lib`, `/usr/local/lib` หรือกำหนด `LD_LIBRARY_PATH`)
- ติดตั้ง Rust (ผ่าน `rustup`) และชุดเครื่องมือคอมไพล์พื้นฐาน (เช่น `build-essential`)

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

หากระบบไม่พบ `libDeckLinkAPI.so` ให้เพิ่มตำแหน่งไลบรารีไปยัง `LD_LIBRARY_PATH` หรือคัดลอกไปยังไดเรกทอรีมาตรฐานก่อนคอมไพล์/รัน

## การใช้งาน (ตัวอย่างไบนารี)
แอปตัวอย่างทุกตัวในตอนนี้เลือกใช้อุปกรณ์ตัวแรก (index 0) หากต้องการเลือกอุปกรณ์อื่น ให้แก้ไขโค้ดที่เกี่ยวข้องในไฟล์ตัวอย่างนั้นๆ

- แสดงรายชื่ออุปกรณ์
```bash
cargo run --bin devicelist
```

- พรีวิวผ่าน OpenGL (DeckLink GL helper)
```bash
cargo run --bin capture_preview_gl
```

- ตรวจสอบเฟรม YUV 8-bit บนคอนโซล
```bash
cargo run --bin capture_yuv_dump
```

- ลูปจับภาพแบบง่ายสำหรับทดลอง
```bash
cargo run --bin capture_test
```
กด Esc เพื่อปิดหน้าต่าง/ออกจากโปรแกรมในโหมดพรีวิว

## รายละเอียดด้านเทคนิคโดยย่อ
- C++ shim (`shim/shim.cpp`) ห่อ DeckLink SDK เป็น C ABI ที่ Rust เรียกใช้ง่าย (เช่น `decklink_list_devices`, `decklink_capture_open`, `decklink_capture_get_frame`, `decklink_capture_close`, รวมถึงฟังก์ชัน DeckLink GL preview)
- กำหนดให้ DeckLink ส่งเฟรม YUV 8-bit (UYVY) โดยตรง ตัว shim ทำหน้าที่คัดลอกข้อมูลรายแถวเข้าสู่บัฟเฟอร์ที่ฝั่ง Rust อ่านต่อได้เลย
- `build.rs` จะ:
  - เพิ่มเส้นทางลิงก์แบบ native และลิงก์กับ `libDeckLinkAPI`
  - คอมไพล์ `shim/shim.cpp` และแนบ `DeckLinkAPIDispatch.cpp` หากพบในตำแหน่งที่รู้จัก (`include/`, `DECKLINK_SDK_DIR`, ฯลฯ)

## การแก้ไขปัญหา (Troubleshooting)
- คอมไพล์/รัน แล้วไม่เจอ `libDeckLinkAPI.so`
  - ตรวจสอบว่าติดตั้ง Blackmagic Desktop Video แล้ว และเพิ่มเส้นทางไลบรารีลงใน `LD_LIBRARY_PATH` หรือคัดลอก `.so` ไปยัง `/usr/lib` / `/usr/local/lib`
- คอมไพล์ไม่พบ `DeckLinkAPIDispatch.cpp`
  - โปรเจ็กต์นี้มีไฟล์ใน `include/` แล้ว หากย้ายออก ให้กำหนดตัวแปร `DECKLINK_SDK_DIR` ให้อยู่ถูกที่ หรือคัดลอกไฟล์กลับมา
- รันแล้วไม่เห็นภาพ
  - ตรวจสอบสัญญาณอินพุตที่เข้าการ์ด DeckLink และสายสัญญาณ
  - ลองใช้ `cargo run --bin devicelist` เพื่อตรวจสอบว่าเห็นอุปกรณ์
  - ถ้าต่อหลายการ์ด/หลายพอร์ต ให้แก้ไข index อุปกรณ์ในซอร์สไบนารีที่ใช้งาน

## ข้อจำกัดปัจจุบัน
- โฟกัสที่ Linux เป็นหลักในตอนนี้
- ยังไม่ได้ทำ CLI เลือกอุปกรณ์/โหมดรับสัญญาณ (แก้ชั่วคราวได้ในซอร์สของไบนารี)
- ยังไม่รองรับการบันทึกเสียง/ไฟล์ในตัวอย่าง

## เครดิตและสิทธิ์การใช้งาน
- DeckLink SDK และไฟล์ที่เกี่ยวข้องเป็นลิขสิทธิ์ของ Blackmagic Design Pty. Ltd. โปรดปฏิบัติตามเงื่อนไขใบอนุญาตของ SDK
- โค้ดตัวอย่างในรีโปนี้ไม่มีการระบุไลเซนส์แยก หากต้องการระบุไลเซนส์ โปรดเพิ่มตามความต้องการของโปรเจ็กต์
