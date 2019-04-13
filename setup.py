from cx_Freeze import setup, Executable

base = None

executables = [Executable("RForest.py", base=base)]

packages = ["idna","numpy","matplotlib","pandas","sklearn"]
options = {
    'build_exe': {
        'packages':packages,
    },
}

setup(
    name = "Random Forest",
    options = options,
    version = "1.26",
    description = 'Random Forest',
    executables = executables
)