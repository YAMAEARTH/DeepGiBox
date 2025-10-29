# 🚀 DeckLink Shim Integration Guide

## ไฟล์ที่ต้องใช้

### ✅ ไฟล์หลักที่ต้องใช้

```
📁 โปรเจคคุณ/
├── shim/
│   ├── shim.cpp                   ⭐ C++ implementation
│   └── decklink.h                 ⭐ C header for Rust FFI
│
├── src/
│   └── decklink.rs                ⭐ Rust bindingson Guide - วิธีเอา Extended Shim ไปใช้ใน Project

## ไฟล์ที่ต้องใช้

### ✅ ไฟล์หลักที่ต้องเอาไป

```
📁 โปรเจคคุณ/
├── shim/
│   ├── shim_extended.cpp          ⭐ ไฟล์นี้ต้องใช้!
│   └── decklink_extended.h        ⭐ ไฟล์นี้ต้องใช้! (สำหรับ Rust FFI)
│
├── src/
│   └── decklink_extended.rs       ⭐ ไฟล์นี้ต้องใช้! (Rust bindings)
│
└── include/                       ✅ มีอยู่แล้ว (DeckLink SDK headers)
    ├── DeckLinkAPI.h
    ├── DeckLinkAPIConfiguration.h
    ├── DeckLinkAPIModes.h
    └── ... (headers อื่นๆ)
```

**หมายเหตุ:** ไฟล์เอกสารและไฟล์ที่ไม่จำเป็นถูกลบออกแล้ว เหลือแค่ไฟล์สำคัญ 3 ไฟล์ด้านบน

---

## 🔧 วิธี Build และใช้งาน

### Build Library

```bash
# Build shim.cpp
g++ -c shim.cpp \
    -I../include \
    -std=c++14 \
    -fPIC \
    -o shim.o

# Link เป็น shared library
g++ shim.o \
    -shared \
    -o libdecklink.so
```

---

## 📝 ตัวอย่างการใช้งานใน Rust

### 1. เพิ่ม dependencies ใน Cargo.toml

```toml
[dependencies]
libc = "0.2"

[build-dependencies]
cc = "1.0"
```

### 2. สร้าง build.rs

```rust
fn main() {
    // Link กับ library ที่ build แล้ว
    println!("cargo:rustc-link-lib=decklink");
    println!("cargo:rustc-link-search=native=./shim");
    
    // หรือถ้า install system-wide
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
```

### 3. ใช้งานใน Rust

```rust
// Import module
mod decklink;
use decklink::*;

fn main() -> Result<(), String> {
    // List devices
    let devices = DeckLinkDevices::new();
    println!("Found {} devices:", devices.count());
    for (i, name) in devices.iter().enumerate() {
        println!("  [{}] {}", i, name);
    }

    // Get device 0
    let device = DeckLinkDevice::new(0);
    
    // Check keying support
    let attrs = device.get_attributes()?;
    if !attrs.supports_internal_keying {
        return Err("Device doesn't support internal keying!".into());
    }
    
    // Enable internal keying
    device.enable_internal_keying()?;
    device.set_keyer_level(255)?;
    
    println!("✅ Internal keying enabled!");
    
    Ok(())
}
```

---

## 🎯 Use Cases ที่พบบ่อย

### Use Case 1: Setup 1080p60 Output with Keying

```rust
use decklink::*;

fn setup_1080p60_with_keying() -> Result<(), String> {
    let device = DeckLinkDevice::new(0);
    
    // 1. Check capabilities
    let attrs = device.get_attributes()?;
    assert!(attrs.supports_internal_keying);
    
    // 2. Open output (จาก shim.cpp เดิม)
    unsafe {
        if !decklink_output_open(0, 1920, 1080, 60.0) {
            return Err("Failed to open output".into());
        }
    }
    
    // 3. Enable keying (จาก shim_extended.cpp ใหม่)
    device.enable_internal_keying()?;
    device.set_keyer_level(255)?;
    
    // 4. Start playback
    unsafe {
        if !decklink_output_start_scheduled_playback() {
            return Err("Failed to start playback".into());
        }
    }
    
    println!("✅ 1080p60 output with keying ready!");
    Ok(())
}
```

### Use Case 2: Query Available Display Modes

```rust
fn list_display_modes() {
    let modes = DisplayModes::new(0);
    
    println!("Available display modes:");
    for (i, mode) in modes.iter().enumerate() {
        println!("  [{}] {} - {}x{} @ {:.2} fps",
            i, mode.name, mode.width, mode.height, mode.fps);
    }
}
```

### Use Case 3: Fade Effects

```rust
fn fade_effect() -> Result<(), String> {
    let device = DeckLinkDevice::new(0);
    
    // Enable keying
    device.enable_internal_keying()?;
    
    // Fade in over 60 frames (1 second @ 60fps)
    device.fade_in(60)?;
    std::thread::sleep(Duration::from_secs(5));
    
    // Fade out
    device.fade_out(60)?;
    std::thread::sleep(Duration::from_secs(1));
    
    // Disable
    device.disable_keying()?;
    
    Ok(())
}
```

---

## 📋 Checklist การ Integration

### สำหรับ C++ Side

- [ ] คัดลอก `shim_extended.cpp` ไป
- [ ] คัดลอก `decklink_extended.h` ไป
- [ ] แก้ include paths ให้ชี้ไปยัง DeckLink SDK headers
- [ ] Build เป็น shared library (.so)
- [ ] ทดสอบ link ได้

### สำหรับ Rust Side

- [ ] คัดลอก `decklink_extended.rs` ไป
- [ ] สร้าง `build.rs` สำหรับ link library
- [ ] เพิ่ม dependencies ใน `Cargo.toml`
- [ ] ทดสอบเรียกใช้ฟังก์ชันได้

---

## 🔍 ตรวจสอบว่าใช้งานได้

### Test 1: List Devices

```rust
#[test]
fn test_list_devices() {
    let devices = DeckLinkDevices::new();
    assert!(devices.count() > 0, "No DeckLink devices found!");
    
    for (i, name) in devices.iter().enumerate() {
        println!("[{}] {}", i, name);
    }
}
```

### Test 2: Check Keying Support

```rust
#[test]
fn test_keying_support() {
    let device = DeckLinkDevice::new(0);
    let attrs = device.get_attributes().unwrap();
    
    println!("Internal keying: {}", attrs.supports_internal_keying);
    println!("External keying: {}", attrs.supports_external_keying);
}
```

### Test 3: Enable Keying

```rust
#[test]
fn test_enable_keying() {
    let device = DeckLinkDevice::new(0);
    
    // This should work if device supports keying
    if let Ok(_) = device.enable_internal_keying() {
        println!("✅ Keying enabled successfully");
        device.set_keyer_level(255).unwrap();
        device.disable_keying().unwrap();
    }
}
```

---

## 🐛 Troubleshooting

### ปัญหา: Library not found

```bash
# Linux: เพิ่ม library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./shim

# หรือ install system-wide
sudo cp libdecklink.so /usr/local/lib/
sudo ldconfig
```

### ปัญหา: Undefined symbols

```bash
# ตรวจสอบ symbols ที่มี
nm -D libdecklink.so | grep decklink

# ควรเห็นฟังก์ชันเหล่านี้:
# decklink_list_devices
# decklink_keyer_enable_internal
# decklink_get_device_attributes
# etc.
```

### ปัญหา: Compilation errors

```bash
# ตรวจสอบ include paths
g++ -c shim.cpp -I../include -v

# ตรวจสอบว่ามี headers ที่ต้องการ
ls ../include/DeckLinkAPI*.h
```

---

## 📚 ไฟล์ในโฟลเดอร์

```
shim/
├── shim.cpp      # C++ implementation with all DeckLink features
├── decklink.h    # C header for Rust FFI bindings
├── decklink.rs   # Rust safe wrappers and types
└── README.md     # Integration guide (this file)
```

---

## ✅ สรุป

**ไฟล์ที่มีในโฟลเดอร์ shim:**

1. ⭐ **shim.cpp** - C++ implementation with full DeckLink control
2. ⭐ **decklink.h** - C header for Rust FFI
3. ⭐ **decklink.rs** - Rust safe wrappers
4. 📖 **README.md** - Integration guide

**Build แล้วได้ shared library:**
- `libdecklink.so`

**ใช้ใน Rust โดย:**
- Link กับ `libdecklink.so`
- Import `mod decklink`
- เรียกใช้ฟังก์ชันผ่าน safe wrappers

🎉 **พร้อมใช้งาน!**
