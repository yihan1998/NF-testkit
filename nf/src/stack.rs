#[derive(Copy, Clone)]
pub enum Stack {
    Empty,

    DataPlaneTcpIpStack,
    LinuxTcpIp,

    DataPlaneGrpc,
    LinuxGrpc,
}
impl Stack {
    pub fn run(
        &self,
        pkt: * mut u8,
        len: i32,
    ) -> bool {
        return true;
    }
}