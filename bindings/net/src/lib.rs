#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::c_int;

fn convert_error(ret: c_int) -> Result<(), i32> {
    if ret == 0 {
        Ok(())
    } else {
        Err(ret as i32)
    }
}
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub fn net_init(nb_core: i32, nb_rxq: i32, nb_txq: i32) -> Result<(), i32> {
    convert_error(unsafe {
        ffi::net_init(nb_core as c_int, nb_rxq as c_int, nb_txq as c_int)
    })
}

pub fn net_setup(lcore_id: i32) -> Result<(), i32> {
    convert_error(unsafe {
        ffi::net_setup(lcore_id as c_int)
    })
}

pub fn net_rx() -> i32 {
    unsafe { ffi::net_tx() }
}

pub fn net_get_rxpkt(pkt_len: *mut i32) -> *mut u8 {
    unsafe { ffi::net_get_rxpkt(pkt_len) }
}

pub fn net_tx() -> i32 {
    unsafe { ffi::net_tx() }
}

pub fn net_get_txpkt(pkt_len: i32) -> *mut u8 {
    unsafe { ffi::net_get_txpkt(pkt_len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = net_init(1,1,1);
        assert_eq!(result, Err(-1));
    }
}
