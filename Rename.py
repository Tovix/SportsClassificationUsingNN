import re
import os
import shutil
import sys
import time

TEST_PATH = "Test"
LABELED_TEST_PATH = "Test/Labeled Test"
TEAM_RANGES = {"Yousif": range(0, 100), "Karim": range(100, 200), "Yossry": range(200, 300),
               "Mostafa": range(300, 400), "Lara": range(400, 500), "Yousif's Bonus": range(500, 538),
               "Karim's Bonus": range(538, 576), "Yossry's Bonus": range(576, 613),
               "Mostafa's Bonus": range(613, 651), "Lara's Bonus": range(651, 688)}
CLASSES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]


def printUsersMenu(error: bool = False):
    os.system("cls")
    if error:
        print(f"Error, your input must be a number in the range [1, {len(TEAM_RANGES)}].")
        print("You are exhausting my patience :(")
    else:
        print("Follow the rules, or you will regret it :)\n")
        print(f"You must enter a number in the range [1, {len(TEAM_RANGES)}]")
        print("------------------------------------------------")
    keys = list(TEAM_RANGES.keys())
    for i in range(len(keys)):
        print(f"{i + 1}- {keys[i]}")


def printImageMenu(error: bool = False):
    os.system("cls")
    if error:
        print(f"Error, your input must be a number in the range [1, {len(CLASSES)}].")
    else:
        print(f"You must enter a number in the range [1, {len(CLASSES)}]")
        print("------------------------------------------------")
    print("MCQ :) what is the previous image class ?")
    for i in range(len(CLASSES)):
        print(f"{i + 1}- {CLASSES[i]}")


def printUsersMessage():
    os.system("cls")
    if(user <= 5):
        if user % 5 == 0:
            print("!!! انا تبيعينى بفطيرة وتعملى فيا المقلب")
            print("دلوقتى يحرقلك ام الجهاز stack overflow دة انا هظرفك")

        # Yousif
        elif user % 5 == 1:
            print("No message for you Sage")

        # Karim
        elif user % 5 == 2:
            print("IOT يحرق ميتين ال")

        # Yossry
        elif user % 5 == 3:
            print("الكسول بتاعنا")

        # Mostafa
        elif user % 5 == 4:
            print("!!! وبتعمل ايه os انت سايب ال")

        print("\nYou have exhausted my patience")
        for remaining in range(15, -1, -1):
            sys.stdout.write("\r")
            sys.stdout.write(f"{remaining} التدمير الذاتى ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r")
        sys.stdout.write("Congratulations!!! you have completed your work\n")
        print("We will appreciate it if you completed your bonus either :)")

    else:
        print("Nothing is perfect, but this was perfection <3\n")


def getUserChoice(printingCommand, iterableObject, error: bool = False):
    printingCommand(error)
    userInput = input()
    search = re.search("\D+", userInput)
    if search or userInput == '' or int(userInput) not in range(1, len(iterableObject) + 1):
        return getUserChoice(printingCommand, iterableObject, True)
    return int(userInput)


def getPrevLabeledImages():
    images = []
    if not os.path.exists(LABELED_TEST_PATH):
        os.mkdir(LABELED_TEST_PATH)
    keys = list(TEAM_RANGES.keys())
    for image in os.listdir(LABELED_TEST_PATH):
        imageID = int(re.search("(\d+).*", image).group(1))
        if imageID in TEAM_RANGES[keys[user - 1]]:
            images.append(imageID)
    return images


def rename():
    keys = list(TEAM_RANGES.keys())
    imagesRange = TEAM_RANGES[keys[user - 1]]
    for image in os.listdir(TEST_PATH):
        imageID = int(re.search("(\d+).*", image).group(1))
        # If the image belongs to the current user
        if imageID in imagesRange and imageID not in prevLabeledImages:
            # open image
            path = TEST_PATH + "\\" + str(imageID) + '.jpg'
            os.system(path)
            classIndex = getUserChoice(printImageMenu, CLASSES) - 1

            # copy and rename image
            newPath = LABELED_TEST_PATH + "\\" + CLASSES[classIndex] + "_" + str(imageID) + '.jpg'
            shutil.copyfile(path, newPath)


user = getUserChoice(printUsersMenu, TEAM_RANGES)
prevLabeledImages = getPrevLabeledImages()
rename()
printUsersMessage()
os.system("pause")
