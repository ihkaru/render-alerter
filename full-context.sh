# Define the path to your project's root folder.
# Using "." means the script will run in the directory where you execute it.
$projectPath = "."

# Define the output file name
$outputFile = "full_context.txt"
$fullOutputPath = Join-Path $projectPath $outputFile

# Define the system requirements file path
$requirementsFile = Join-Path $projectPath "system-requirement.md"

# Check if the system requirements file exists before trying to read it.
if (Test-Path $requirementsFile) {
    # Start with the system requirements file
    Get-Content $requirementsFile | Out-File $fullOutputPath
} else {
    # If the file doesn't exist, create an empty output file and show a warning.
    Write-Warning "Warning: system-requirement.md not found."
    Clear-Content $fullOutputPath
}

# Define the code file extensions you want to include.
$extensions = @("*.py", "*.json", "*.yml", "*.txt", "dockerfile") # <-- EDIT THESE EXTENSIONS

# Find all specified files recursively.
Get-ChildItem -Path $projectPath -Recurse -Include $extensions | ForEach-Object {
    # Exclude the output file itself from being added to the context.
    if ($_.FullName -ne (Resolve-Path $fullOutputPath)) {
        Add-Content $fullOutputPath "`n--- File: $($_.FullName) ---`n"
        # Use the -Raw parameter to read the entire file content at once, which is more efficient.
        Add-Content $fullOutputPath (Get-Content $_.FullName -Raw)
    }
}

Write-Host "Done! Project context has been saved to $fullOutputPath"