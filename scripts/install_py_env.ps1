## ------------------------------------------------
## set env variables
## ------------------------------------------------
$pycodepath = "."
$activateEnvPs1 = "$pycodepath\env\Scripts\Activate.ps1"


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
    python -m pip install --upgrade pip
    python -m pip install -U virtualenv
    python -m virtualenv venv
    echo "python -m venv $pycodepath\env"
    python -m venv $pycodepath\env
}

## ------------------------------------------------
## Activate Python venv
## ------------------------------------------------
Set-ExecutionPolicy Unrestricted -Scope Process
. $activateEnvPs1

echo $(pip list)
echo $(cat "$pycodepath\requirements.txt")
echo $("$pycodepath\requirements.txt")
echo "Python packets will be intalled: $(cat "$pycodepath\requirements.txt")"

python -m pip install --upgrade pip
python -m pip install -r "$pycodepath\requirements.txt"
