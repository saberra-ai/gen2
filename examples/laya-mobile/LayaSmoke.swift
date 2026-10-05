import Foundation

// Include gen2_laya.h in the application's bridging header.
// Call open/decide/resume/close from a background queue. suspend is nonblocking.
enum LayaSmoke {
    private static func consume(_ pointer: UnsafeMutablePointer<CChar>?) throws -> String {
        guard let pointer else { throw CocoaError(.coderInvalidValue) }
        defer { gen2_laya_free(pointer) }
        return String(cString: pointer)
    }
    static func open(configJSON: String) throws -> String {
        try configJSON.withCString { try consume(gen2_laya_open($0)) }
    }
    static func decide(handle: UInt64, requestJSON: String) throws -> String {
        try requestJSON.withCString { try consume(gen2_laya_decide(handle, $0)) }
    }
    static func suspend(handle: UInt64) throws -> String {
        try consume(gen2_laya_suspend(handle))
    }
    static func invoke(handle: UInt64, invocationJSON: String) throws -> String {
        try invocationJSON.withCString { try consume(gen2_laya_invoke(handle, $0)) }
    }
    static func resume(handle: UInt64) throws -> String {
        try consume(gen2_laya_resume(handle))
    }
    static func close(handle: UInt64) throws -> String {
        try consume(gen2_laya_close(handle))
    }
}
