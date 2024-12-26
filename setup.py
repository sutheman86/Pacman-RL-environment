import os
import subprocess
import sys
import shutil

bold_ch = '\033[1m'
red_ch = '\033[91m'
end_ch = '\033[0m'

def print_error(str):
    print(bold_ch + red_ch + str + end_ch)
def print_bold(str):
    print(bold_ch + str + end_ch)

with_grid = 0
with_advanced_reward = 0
project_root = os.getcwd()
gymnasium_dir_entry = os.path.join(project_root, "gymnasium")
src_dir = os.path.join(project_root, "src")
nb_dir = os.path.join(project_root, "notebook")
check_exist = [gymnasium_dir_entry, src_dir, project_root, nb_dir]

for path in check_exist:
    if not os.path.exists:
        print_error("Necessary path: " + path + " does not exist, check your installation!")

dir = ""

with_grid = input(bold_ch + "State Representation: Grid or Pixel? (1 = Grid, 2 = Pixel): " + end_ch)
if with_grid == "1":
    dir = dir + "grid"
elif with_grid == "2":
    dir = dir + "pixel"
else:
    print_error("Invalid Option!!")
    exit(1)

with_advanced_reward = input(bold_ch + "Include advanced reward mechanism: Yes or no? (1 = Yes, 2 = No): " + end_ch)
if with_advanced_reward == "1":
    dir = dir + "_advanced"
elif with_advanced_reward == "2":
    dir = dir + "_advanced"
else:
    print_error("Invalid value!!")
    exit(1)

gymnasium_dir_entry = os.path.join(gymnasium_dir_entry, dir)
print(gymnasium_dir_entry)

src_source = ""
nb_source = ""
if with_grid == "1":
    src_source = os.path.join(src_dir, "grid")
    nb_source = os.path.join(nb_dir, "grid")
else:
    src_source = os.path.join(src_dir, "pixel")
    nb_source = os.path.join(nb_dir, "pixel")

src_bak = os.path.join(src_dir, "old")
nb_bak = os.path.join(nb_dir, "old")

if not os.path.exists(src_bak):
    os.makedirs(src_bak)
if not os.path.exists(nb_bak):
    os.makedirs(nb_bak)

for fname in os.listdir(src_dir):
    if fname.endswith(".py"):
        srcf = os.path.join(src_dir, fname)
        dstf = os.path.join(src_bak, fname)
        shutil.move(srcf, dstf)

for fname in os.listdir(nb_dir):
    if fname.endswith(".ipynb"):
        srcf = os.path.join(nb_dir, fname)
        dstf = os.path.join(nb_bak, fname)
        shutil.move(srcf, dstf)

print_error("Check \"old\" directory in /src and /notebook for old files!")

shutil.copytree(src_source, src_dir, dirs_exist_ok=True)
shutil.copytree(nb_source, nb_dir, dirs_exist_ok=True)

if not os.path.exists(gymnasium_dir_entry):
    print_error("Error: Check your installation!" )
else:
    print_bold("Installing gymnasium environment as editable package...")
    if subprocess.check_call(["pip", "install", "-e", gymnasium_dir_entry]):
        print_error("Installation Failed :(")
    else:
        print_bold("Installation done! Check \"gymnasium/" + dir + "\" for gymnasium environment and game code")


