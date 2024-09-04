Scriptname SPI_DDAScript extends ReferenceAlias  
Import MiscUtil ; Import correct utilities for general functions

; Declare arrays with the correct syntax
float[] npcDamageTaken
float[] playerDamageTaken

int currentDifficultyIndex = 2 ; Start at Adept

Event OnInit()
    ; Initialize arrays with the correct size
    npcDamageTaken = new float[6]
    playerDamageTaken = new float[6]

    ; Set values for the NPC damage multiplier array
    npcDamageTaken[0] = 2.0    ; Novice
    npcDamageTaken[1] = 1.5    ; Apprentice
    npcDamageTaken[2] = 1.0    ; Adept (Default)
    npcDamageTaken[3] = 0.75   ; Expert
    npcDamageTaken[4] = 0.5    ; Master
    npcDamageTaken[5] = 0.25   ; Legendary

    ; Set values for the Player damage multiplier array
    playerDamageTaken[0] = 0.5  ; Novice
    playerDamageTaken[1] = 0.75 ; Apprentice
    playerDamageTaken[2] = 1.0  ; Adept (Default)
    playerDamageTaken[3] = 1.5  ; Expert
    playerDamageTaken[4] = 2.0  ; Master
    playerDamageTaken[5] = 3.0  ; Legendary

    RegisterForSingleUpdate(1.0) ; Start the update loop
EndEvent

Event OnUpdate()
    ; Specify the path to your text file
    String filePath = "emotion_results.txt"
    ; Read the file content
    String difficultyLevelStr = MiscUtil.ReadFromFile(filePath)

    ; Debug.Notification("current emotion: " + difficultyLevelStr)
    ; Debug.Notification("current difficulty index: " + currentDifficultyIndex)
    ; Debug.Notification("fDiffMultHPByPCN setting currently: " + Game.GetGameSettingFloat("fDiffMultHPByPCN"))
    ; Debug.Notification("fDiffMultHPToPCN setting currently: " + Game.GetGameSettingFloat("fDiffMultHPToPCN"))

    ; Adjust difficulty based on input
    If difficultyLevelStr == "neutral"
        ;dont change difficulty
        currentDifficultyIndex = currentDifficultyIndex 

    ElseIf difficultyLevelStr == "angry"
        ;large decrease difficulty but ensure it doesn't go below Novice (index 0)
        currentDifficultyIndex = currentDifficultyIndex - 2
        if currentDifficultyIndex < 0
            currentDifficultyIndex = 0
        endif
    ElseIf difficultyLevelStr == "happy"
        ;lage increase difficulty but ensure it doesn't go above Legendary (index 5)
        currentDifficultyIndex = currentDifficultyIndex + 2
        if currentDifficultyIndex > 5
            currentDifficultyIndex = 5
        endif
    ElseIf difficultyLevelStr == "fear"
        ;slight decrease difficulty but ensure it doesn't go below Novice (index 0)
        currentDifficultyIndex = currentDifficultyIndex - 1
        if currentDifficultyIndex < 0
            currentDifficultyIndex = 0
        endif
    ElseIf difficultyLevelStr == "sad"
        ; slightly decrease difficulty but ensure it doesn't go above Legendary (index 5)
        currentDifficultyIndex = currentDifficultyIndex - 1
        if currentDifficultyIndex < 0
            currentDifficultyIndex = 0
        endif
    ElseIf difficultyLevelStr == "surprise"
        ; slight ncrease difficulty but ensure it doesn't go above Legendary (index 5)
        currentDifficultyIndex = currentDifficultyIndex + 1
        if currentDifficultyIndex > 5
            currentDifficultyIndex = 5
        endif
    ElseIf difficultyLevelStr == "disgust"
        ; increase decrease difficulty but ensure it doesn't go above Legendary (index 5)
        currentDifficultyIndex = currentDifficultyIndex + 1
        if currentDifficultyIndex > 5
            currentDifficultyIndex = 5
        endif
    ElseIf difficultyLevelStr == "calm"
        ;dont change difficulty
        currentDifficultyIndex = currentDifficultyIndex 
    EndIf

    ; Update the global variables with the new values from the arrays
    Game.SetGameSettingFloat("fDiffMultHPByPCN", npcDamageTaken[currentDifficultyIndex])
    Game.SetGameSettingFloat("fDiffMultHPToPCN", playerDamageTaken[currentDifficultyIndex])
        

    RegisterForSingleUpdate(5.0) ; Recheck every 5 seconds
EndEvent