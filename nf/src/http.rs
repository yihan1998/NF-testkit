pub enum ApplicationLayerProtocol {
    HttpRequest(HttpRequest),
    HttpResponse(HttpResponse),
}

struct HttpRequest {
    method: HttpMethod,
    uri: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

struct HttpResponse {
    status_code: u16,
    reason_phrase: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

enum HttpMethod {
    GET,
    POST,
    // Other HTTP methods as needed
}