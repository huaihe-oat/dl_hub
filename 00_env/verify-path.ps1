$ErrorActionPreference = 'Continue'

$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User')

Write-Host '=== where.exe python ==='
where.exe python

Write-Host ''
Write-Host '=== which python wins ==='
python --version

Write-Host ''
Write-Host '=== base python ssl (this used to fail) ==='
python -c "import ssl; print('ssl OK:', ssl.OPENSSL_VERSION)"

Write-Host ''
Write-Host '=== pip available? ==='
python -m pip --version

Write-Host ''
Write-Host '=== conda activate dl still works? ==='
conda activate dl
python -c "import sys, torch; print('exe  :', sys.executable); print('torch:', torch.__version__); print('cuda :', torch.cuda.is_available())"
