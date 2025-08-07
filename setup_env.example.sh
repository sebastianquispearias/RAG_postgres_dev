# setup_env.ps1

# Salir al primer error
$ErrorActionPreference = "Stop"

# 1. Eliminar todos los resource-groups que empiecen con "dev"
Write-Host "⏳ Eliminando resource groups dev*..."
Get-AzResourceGroup | Where-Object { $_.ResourceGroupName -like "dev*" } `
  | ForEach-Object {
      Write-Host "  ➜ Eliminando $($_.ResourceGroupName)"
      Remove-AzResourceGroup -Name $_.ResourceGroupName -Force -AsJob
    }

# 2. Limpiar entornos locales de azd
Write-Host "⏳ Limpiando carpeta .azure/..."
Remove-Item -Recurse -Force .azure -ErrorAction SilentlyContinue

# 3. Crear nuevo entorno azd
$envName = "dev_auto"
Write-Host "⏳ Creando entorno azd: $envName"
azd env new $envName

# 4. Definir todas las variables de entorno
$vars = @{
  AZURE_LOCATION           = "centralus"
  AZURE_OPENAI_LOCATION    = "eastus"
  DEPLOY_AZURE_OPENAI      = "false"
  AZURE_OPENAI_ENDPOINT    = "https://rgroup2464-sqa.openai.azure.com/"
  AZURE_OPENAI_KEY         = ""
  OPENAI_CHAT_HOST         = "azure"
  OPENAI_EMBED_HOST        = "azure"
  POSTGRES_HOST            = "zane-linave.postgres.database.azure.com"
  POSTGRES_USERNAME        = "sebastian"
  POSTGRES_PASSWORD        = "Dtj78frWb2"
  POSTGRES_DATABASE        = "dev_linave"
  POSTGRES_SSL             = "require"
  PG_PORT                  = "5432"
  USE_AI_PROJECT           = "false"
  DEPLOY_EVAL_MODEL        = "false"
}

Write-Host "⏳ Aplicando variables a azd..."
foreach ($key in $vars.Keys) {
  azd env set $key $vars[$key]
}

# 5. Mostrar valores aplicados
Write-Host "`n✅ Valores de entorno:"
azd env get-values

Write-Host "`n✅ ¡Listo! Ahora ejecuta:`n    azd up"
