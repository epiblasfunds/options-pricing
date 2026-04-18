resource "aws_dynamodb_table" "dynamo_table" {
  name           = var.table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "pk"  # partition key
  range_key      = "sk"  # sort key

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }
}
