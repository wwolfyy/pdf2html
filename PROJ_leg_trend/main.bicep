resource appServicePlan 'Microsoft.Web/serverfarms@2021-02-01' = {
  name: 'appServicePlanName'
  location: 'location'
  sku: {
    name: 'B1'
    tier: 'Basic'
  }
  properties: {
    numberOfWorkers: 1
  }
}

resource webApp 'Microsoft.Web/sites@2021-02-01' = {
  name: 'webAppName'
  location: 'location'
  properties: {
    serverFarmId: appServicePlan.id
  }
}
