@echo off
echo Pushing Crop Yield API to GitHub (https://github.com/Premaben07/crop-yield-api)...
git push -u origin main
if %errorlevel% equ 0 (
    echo.
    echo ==============================================
    echo Successfully pushed to GitHub!
    echo Visit: https://github.com/Premaben07/crop-yield-api
    echo ==============================================
) else (
    echo.
    echo ==============================================
    echo Push encountered an issue.
    echo Make sure you have created the repository at:
    echo https://github.com/new (Name: crop-yield-api)
    echo ==============================================
)
pause
