import Foundation

/// A job/task posted by a lab (the `tasks` table). Contributors record data for it.
struct Job: Identifiable, Decodable, Hashable {
    let id: UUID
    let title: String
    let description: String?
    let dataType: String?
    let requiredCapabilities: [String]?
    let bountyAmount: Double?
    let quantityNeeded: Int?
    let quantityFilled: Int?
    let status: String?

    enum CodingKeys: String, CodingKey {
        case id, title, description, status
        case dataType = "data_type"
        case requiredCapabilities = "required_capabilities"
        case bountyAmount = "bounty_amount"
        case quantityNeeded = "quantity_needed"
        case quantityFilled = "quantity_filled"
    }
}
