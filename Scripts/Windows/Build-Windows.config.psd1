@{
  Build = @{
    WorkspaceRootEnv = 'WORKSPACE_PATH'
    LogDir = 'logs/windows'

    CargoTargetDir = 'target'
    CargoFeatures = @()
  }

  Msix = @{
    PackageName = 'Kataglyphis.RustProjectTemplate'
    Publisher = 'CN=Kataglyphis'
    PublisherDisplayName = 'Kataglyphis'
    DisplayName = 'Kataglyphis Rust Project Template'
    Description = 'Rust project template with optional GUI, ONNX backends, profiling, packaging, and CI workflows.'
    Version = '0.1.0.0'
    MinVersion = '10.0.19041.0'
    ManifestTemplate = 'Scripts/Windows/AppxManifest.xml.template'
    Binary = 'kataglyphis_rustprojecttemplate'
  }
}