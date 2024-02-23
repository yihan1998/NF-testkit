use crate::network;

use std::slice;

#[derive(Copy, Clone)]
pub enum Stack {
    Empty,  // Used for packet-based test
    FullStack,
}
impl Stack {
    pub fn run(
        &self,
        pkt: *const u8,
        len: usize,
    ) -> Result<(), i32> {
        unsafe {
            let data: Vec<u8> = slice::from_raw_parts(pkt, len).to_vec();
            return network::handle_packet(data); 
        }
    }
}