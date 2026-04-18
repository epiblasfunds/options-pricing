resource "aws_ecr_repository" "volatility_repo" {
  name = "lambda_volatility"
}

data "aws_s3_bucket" "tmp_db" {
  bucket = var.bucket_name
}

resource "aws_lambda_function" "dynamo_writer" {
  function_name   = "writer_mini_ibex_options"
  image_uri       = "${aws_ecr_repository.volatility_repo.repository_url}:latest"
  package_type    = "Image"
  memory_size     = 256
  timeout         = 600
  role            = aws_iam_role.lambda_exec_volatility.arn

  environment {
    variables = {
      TABLE_NAME  = var.table_name
    }
  }
}

resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dynamo_writer.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = data.aws_s3_bucket.tmp_db.arn
}

resource "aws_s3_bucket_notification" "trigger_lambda_on_object" {
  bucket = data.aws_s3_bucket.tmp_db.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.dynamo_writer.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = ""  
    filter_suffix       = ""
  }

  depends_on = [aws_lambda_permission.allow_s3]
}

resource "aws_iam_role" "lambda_exec_volatility" {
  name = "lambda_execution_role_volatility"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_policy" "lambda_dynamo_logs_policy_volatility" {
  name        = "lambda-dynamo-logs-policy-volatility"
  description = "Permite a Lambda escribir en CloudWatch Logs y acceder a DynamoDB"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = [
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current_volatility.account_id}:log-group:/aws/lambda/${aws_lambda_function.dynamo_writer.function_name}:*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource = [
          "arn:aws:s3:::mini-ibex-options-tmp-db",
          "arn:aws:s3:::mini-ibex-options-tmp-db/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:GetItem",
          "dynamodb:Scan",
          "dynamodb:Query"
        ],
        Resource = "arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current_volatility.account_id}:table/${var.table_name}"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_volatility" {
  role       = aws_iam_role.lambda_exec_volatility.name
  policy_arn = aws_iam_policy.lambda_dynamo_logs_policy_volatility.arn
}

data "aws_caller_identity" "current_volatility" {}
