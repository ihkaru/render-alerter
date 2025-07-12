# Define the output file name
$outputFile = "full_context.txt"

# Start with the system requirements file
Get-Content system-requirement.md | Out-File $outputFile

# Define the code file extensions you want to include
$extensions = @("*.py", "*.json", "*.yml", "*.txt", "dockerfile") # <-- EDIT THESE EXTENSIONS

# Find all code files, and append their name and content to the output file
Get-ChildItem -Recurse -Include $extensions | ForEach-Object {
    Add-Content $outputFile "`n--- File: $($_.FullName) ---`n"
    Add-Content $outputFile (Get-Content $_.FullName)
}

Write-Host "Done! Project context has been saved to full_context.txt"