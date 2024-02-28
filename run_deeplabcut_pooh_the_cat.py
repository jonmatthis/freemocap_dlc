# %% import stuff
import deeplabcut

# %% create project
response = deeplabcut.create_pretrained_project(
    "pooh-the-cat",
    "jsm",
    [r"C:\Users\jonma\Sync\videos\2024-02-28_pooh_gopros\GX010034_clip1.mp4"],
    model="full_cat",
    working_directory=r"D:\deeplabcut-models",
    copy_videos=True,
    videotype=".mp4",
    analyzevideo=True,
    filtered=True,
    createlabeledvideo=True,
    trainFraction=None,
)

print("-----------------------------")
print(f"Created project: \n\n {response}")
print("-----------------------------")

# %%
