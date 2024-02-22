#[derive(Copy, Clone)]
pub enum Function {
    DnsFilter,
    FlowMonitor,
}
impl Function {
    #[warn(dead_code)]
    pub fn process_request(
        &self,
    ) -> bool {
        return true;
    }
}