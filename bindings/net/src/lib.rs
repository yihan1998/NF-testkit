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

pub fn net_init(nr_core: i32, nr_rxq: i32, nr_txq: i32) -> Result<(), i32> {
    convert_error(unsafe {
        ffi::net_init(nr_core as c_int, nr_rxq as c_int, nr_txq as c_int)
    })
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
