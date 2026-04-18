resource "aws_ecr_repository" "fetch_data_repo" {
  name = "lambda_fetch_data"
}

resource "aws_lambda_function" "ibex_scraper" {
  function_name   = "mini_ibex_options_scraper"
  image_uri       = "${aws_ecr_repository.fetch_data_repo.repository_url}:latest"
  package_type    = "Image"
  timeout         = 600
  role            = aws_iam_role.lambda_exec_fetch_data.arn

  environment {
    variables = {
      BUCKET_NAME  = var.bucket_name
    }
  }
}

resource "aws_cloudwatch_event_rule" "daily_trigger" {
  name                = "run-scraper-daily"
  description         = "Ejecutar Lambda mini_ibex_options_scraper de lunes a viernes"
  schedule_expression = "cron(0 20 ? * MON-FRI *)" # Cada día a las 20:00 UTC (22:00 hora peninsular), L-V
}

resource "aws_cloudwatch_event_target" "trigger_lambda" {
  rule      = aws_cloudwatch_event_rule.daily_trigger.name
  target_id = "invoke_lambda"
  arn       = aws_lambda_function.ibex_scraper.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ibex_scraper.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_trigger.arn
}

resource "aws_iam_role" "lambda_exec_fetch_data" {
  name = "lambda_execution_role_fetch_data"

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

resource "aws_iam_policy" "lambda_fetch_data_logs_policy" {
  name        = "lambda-logs-policy-fetch-data"
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
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current_fetch_data.account_id}:log-group:/aws/lambda/${aws_lambda_function.ibex_scraper.function_name}:*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "s3:PutObject"
        ],
        Resource = "arn:aws:s3:::mini-ibex-options-tmp-db/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment" {
  role       = aws_iam_role.lambda_exec_fetch_data.name
  policy_arn = aws_iam_policy.lambda_fetch_data_logs_policy.arn
}

data "aws_caller_identity" "current_fetch_data" {}
