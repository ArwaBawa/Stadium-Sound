using System.Collections.Generic;

[System.Serializable]
public class Landmark
{
    public float x;
    public float y;
    public float z;
    public float visibility;
}

[System.Serializable]
public class FrameData
{
    public int frame;
    public int width;
    public int height;
    public List<Landmark> pose_landmarks;
    public List<Landmark> left_hand_landmarks;
    public List<Landmark> right_hand_landmarks;
}

[System.Serializable]
public class LandmarkDataWrapper
{
    public List<FrameData> frames;
}
