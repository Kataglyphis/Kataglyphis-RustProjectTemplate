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

  Msi = @{
    # Enable/disable MSI packaging
    Enabled = $true
    # Product name shown in installer
    ProductName = 'Kataglyphis Rust Project Template'
    # Manufacturer name
    Manufacturer = 'Kataglyphis'
    # Path to WiX source file (relative to workspace root)
    WxsFile = 'wix/main.wxs'
    # Path to license file (relative to workspace root)  
    LicenseFile = 'wix/License.rtf'
    # Output filename pattern (version will be appended)
    OutputName = 'kataglyphis_rustprojecttemplate'
  }
}