## ------------------------------------------------
## set env variables
## ------------------------------------------------
$pycodepath = "."
$envPath = "$pycodepath\env"


## ------------------------------------------------
## Select Python version
## ------------------------------------------------

# redirect stderr into stdout
$p = &{python -V} 2>&1
# check if an ErrorRecord was returned
$version = if($p -is [System.Management.Automation.ErrorRecord]) { $p.Exception.Message } else { $p }
echo $p
echo $version


## ------------------------------------------------
## Install Python venv
## ------------------------------------------------
if (!(Test-Path -Path "$pycodepath\env\")) {
    conda env create --prefix $envPath -f requirements.yaml
}

## ------------------------------------------------
## Activate Python venv
## ------------------------------------------------
Set-ExecutionPolicy Unrestricted -Scope Process
conda activate $envPath
