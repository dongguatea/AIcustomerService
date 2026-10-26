$ErrorActionPreference = "Stop"
$body = Join-Path $env:TEMP "mewhelp-chat-body.json"
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Send-Turn {
    param([string]$SessionId, [string]$Message)
    $payload = @{ session_id = $SessionId; message = $Message } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($body, $payload, $utf8)
    Write-Host "=== 问: $Message ==="
    curl.exe -sS -N -X POST http://localhost:8000/api/chat `
        -H "Content-Type: application/json" `
        --data-binary "@$body"
    Write-Host ""
}

Send-Turn -SessionId "demo-ps" -Message "我叫王小明,昨天刚买了一台智能猫砂盆"
Send-Turn -SessionId "demo-ps" -Message "还记得我叫什么、买了什么吗?"
