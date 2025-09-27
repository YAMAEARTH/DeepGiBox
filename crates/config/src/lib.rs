use anyhow::Result;
#[derive(Debug)] pub struct AppConfig;
impl AppConfig { pub fn from_file(_p:&str)->Result<Self>{ Ok(Self) } }
