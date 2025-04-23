using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

public class MediaPipeAvatarAnimator : MonoBehaviour
{
    public Animator animator;                  // Humanoid avatar
    public string jsonFileName = "pose_and_hands.json";
    public float frameRate = 30f;              // Playback rate
    [Range(0f,1f)] public float smoothFactor = 0.5f;
    [Range(-0.3f, 0.3f)] public float armSpreadAmount = 0.1f;

    private List<FrameData> frames;
    private int idx = 0;
    private Dictionary<string, HumanBodyBones> boneMap;
    private Dictionary<string,int[]> dirMap;
    private Dictionary<HumanBodyBones,Quaternion> prevRot = new();

    void Start()
    {
        // 1️⃣ Load JSON
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        string json = File.ReadAllText(path);
        frames = JsonConvert.DeserializeObject<List<FrameData>>(json);            // :contentReference[oaicite:6]{index=6}

        // 2️⃣ Setup mappings
        SetupBoneMap();                                                         // :contentReference[oaicite:7]{index=7}
        SetupDirectionMap();

        // 3️⃣ Play
        StartCoroutine(Play());
    }

    IEnumerator Play()
    {
        var wait = new WaitForSeconds(1f/frameRate);
        while(idx < frames.Count)
        {
            ApplyPose(frames[idx].pose_landmarks, "");                           // Pose bones
            ApplyPose(frames[idx].left_hand_landmarks, "left_");                 // Left hand
            ApplyPose(frames[idx].right_hand_landmarks, "right_");               // Right hand
            idx++;
            yield return wait;
        }
    }

    private Dictionary<HumanBodyBones, Vector3> boneForwardAxis = new Dictionary<HumanBodyBones, Vector3>()
    {
        //Left
        { HumanBodyBones.LeftUpperArm, Vector3.down },
        { HumanBodyBones.LeftLowerArm, Vector3.down },
        { HumanBodyBones.LeftHand, Vector3.left },
        { HumanBodyBones.LeftIndexProximal, Vector3.back },
        { HumanBodyBones.LeftIndexIntermediate, Vector3.up },
        { HumanBodyBones.LeftIndexDistal, Vector3.up },
        { HumanBodyBones.LeftMiddleProximal, Vector3.back },
        { HumanBodyBones.LeftMiddleIntermediate, Vector3.up },
        { HumanBodyBones.LeftMiddleDistal, Vector3.up },
        { HumanBodyBones.LeftRingProximal, Vector3.down },
        { HumanBodyBones.LeftRingIntermediate, Vector3.up },
        { HumanBodyBones.LeftRingDistal, Vector3.up },
        { HumanBodyBones.LeftLittleProximal, Vector3.down },
        { HumanBodyBones.LeftLittleIntermediate, Vector3.up },
        { HumanBodyBones.LeftLittleDistal, Vector3.up },
        { HumanBodyBones.LeftThumbProximal, Vector3.forward },
        { HumanBodyBones.LeftThumbIntermediate, Vector3.forward },
        { HumanBodyBones.LeftThumbDistal, Vector3.back },
        
        // Right
        { HumanBodyBones.RightUpperArm, Vector3.down },
        { HumanBodyBones.RightLowerArm, Vector3.down },
        { HumanBodyBones.RightHand, Vector3.right },
        { HumanBodyBones.RightIndexProximal, Vector3.forward },
        { HumanBodyBones.RightIndexIntermediate, Vector3.up },
        { HumanBodyBones.RightIndexDistal, Vector3.up },
        { HumanBodyBones.RightMiddleProximal, Vector3.forward },
        { HumanBodyBones.RightMiddleIntermediate, Vector3.up },
        { HumanBodyBones.RightMiddleDistal, Vector3.up },
        { HumanBodyBones.RightRingProximal, Vector3.forward },
        { HumanBodyBones.RightRingIntermediate, Vector3.up },
        { HumanBodyBones.RightRingDistal, Vector3.up },
        { HumanBodyBones.RightLittleProximal, Vector3.forward },
        { HumanBodyBones.RightLittleIntermediate, Vector3.up },
        { HumanBodyBones.RightLittleDistal, Vector3.up },
        { HumanBodyBones.RightThumbProximal, Vector3.back },
        { HumanBodyBones.RightThumbIntermediate, Vector3.back },
        { HumanBodyBones.RightThumbDistal, Vector3.forward },
    };

    void SetupBoneMap()
    {
        boneMap = new Dictionary<string, HumanBodyBones>{
            // Pose
            {"11",HumanBodyBones.LeftUpperArm}, {"13",HumanBodyBones.LeftLowerArm},
            {"15",HumanBodyBones.LeftHand},     {"12",HumanBodyBones.RightUpperArm},
            {"14",HumanBodyBones.RightLowerArm},{"16",HumanBodyBones.RightHand},

            // Left hand (mp_hands 0–20; prefix "left_")
            {"left_0",HumanBodyBones.LeftHand}, {"left_1",HumanBodyBones.LeftThumbProximal},
            {"left_2",HumanBodyBones.LeftThumbIntermediate}, {"left_3",HumanBodyBones.LeftThumbDistal},
            {"left_5",HumanBodyBones.LeftIndexProximal}, {"left_9",HumanBodyBones.LeftMiddleProximal}, {"left_13",HumanBodyBones.LeftRingProximal}, {"left_17",HumanBodyBones.LeftLittleProximal},
            // {"left_6",HumanBodyBones.LeftIndexIntermediate}, {"left_7",HumanBodyBones.LeftIndexDistal}, 
            // {"left_10",HumanBodyBones.LeftMiddleIntermediate}, {"left_11",HumanBodyBones.LeftMiddleDistal},
            //  {"left_14",HumanBodyBones.LeftRingIntermediate},
            // {"left_15",HumanBodyBones.LeftRingDistal}, 
            // {"left_18",HumanBodyBones.LeftLittleIntermediate}, {"left_19",HumanBodyBones.LeftLittleDistal},

            // Right hand
            {"right_0",HumanBodyBones.RightHand}, {"right_1",HumanBodyBones.RightThumbProximal},
            {"right_2",HumanBodyBones.RightThumbIntermediate}, {"right_3",HumanBodyBones.RightThumbDistal},
            {"right_5",HumanBodyBones.RightIndexProximal}, {"right_9",HumanBodyBones.RightMiddleProximal}, {"right_13",HumanBodyBones.RightRingProximal}, {"right_17",HumanBodyBones.RightLittleProximal},
            //  {"right_6",HumanBodyBones.RightIndexIntermediate},
            // {"right_7",HumanBodyBones.RightIndexDistal}, 
            // {"right_10",HumanBodyBones.RightMiddleIntermediate}, {"right_11",HumanBodyBones.RightMiddleDistal},
            //  {"right_14",HumanBodyBones.RightRingIntermediate},
            // {"right_15",HumanBodyBones.RightRingDistal}, 
            // {"right_18",HumanBodyBones.RightLittleIntermediate}, {"right_19",HumanBodyBones.RightLittleDistal},
        };
    }

    void SetupDirectionMap()
    {
        dirMap = new Dictionary<string,int[]>{
            // Arms
            {"11",new[]{11,13}}, {"13",new[]{13,15}},
            {"12",new[]{12,14}}, {"14",new[]{14,16}},
            // Left hand
            {"left_0",new[]{0,1}}, {"left_1",new[]{1,2}}, {"left_2",new[]{2,3}}, {"left_3",new[]{3,4}},
            {"left_5",new[]{5,6}}, {"left_6",new[]{6,7}}, {"left_7",new[]{7,8}},
            {"left_9",new[]{9,10}},{"left_10",new[]{10,11}},{"left_11",new[]{11,12}},
            {"left_13",new[]{13,14}},{"left_14",new[]{14,15}},{"left_15",new[]{15,16}},
            {"left_17",new[]{17,18}},{"left_18",new[]{18,19}},{"left_19",new[]{19,20}},
            // Right hand (same indices)
            {"right_0",new[]{0,1}},{"right_1",new[]{1,2}},{"right_2",new[]{2,3}},{"right_3",new[]{3,4}},
            {"right_5",new[]{5,6}},{"right_6",new[]{6,7}},{"right_7",new[]{7,8}},
            {"right_9",new[]{9,10}},{"right_10",new[]{10,11}},{"right_11",new[]{11,12}},
            {"right_13",new[]{13,14}},{"right_14",new[]{14,15}},{"right_15",new[]{15,16}},
            {"right_17",new[]{17,18}},{"right_18",new[]{18,19}},{"right_19",new[]{19,20}},
        };
    }

    void ApplyPose(List<Landmark> lm, string prefix)
    {
        if(lm==null) return;
        foreach(var kv in boneMap)
        {
            string key = kv.Key;
            if(!key.StartsWith(prefix)) continue;
            if (!dirMap.ContainsKey(key)) continue;
            int[] p = dirMap[key];
            if(p[0]>=lm.Count||p[1]>=lm.Count) continue;

            // 1️⃣ Compute direction
            Vector3 a = new Vector3(lm[p[0]].x, lm[p[0]].y, lm[p[0]].z);
            Vector3 b = new Vector3(lm[p[1]].x, lm[p[1]].y, lm[p[1]].z);
            // Vector3 dir = (b - a).normalized;                                   // :contentReference[oaicite:8]{index=8}

            Vector3 spreadOffset = Vector3.zero;

            var boneKey = kv.Value;
            if (boneKey == HumanBodyBones.LeftUpperArm || boneKey == HumanBodyBones.LeftLowerArm)
            {
                spreadOffset = Vector3.left * armSpreadAmount;  // Push left arm to the left
            }
            else if (boneKey == HumanBodyBones.RightUpperArm || boneKey == HumanBodyBones.RightLowerArm)
            {
                spreadOffset = Vector3.right * armSpreadAmount; // Push right arm to the right
            }

            var dir = (b - a + spreadOffset).normalized;

            // 2️⃣ Bone & its forward axis (assume Z+; adjust per model)
            var bone = animator.GetBoneTransform(kv.Value);
            if(bone==null) continue;
            var forwadAxis = boneForwardAxis.ContainsKey(kv.Value) ? boneForwardAxis[kv.Value] : Vector3.forward;
            // var forwadAxis = Vector3.down;
            Vector3 curr = bone.rotation * forwadAxis;

            // 3️⃣ Rotation delta & apply
            Quaternion delta = Quaternion.FromToRotation(curr, dir);            // :contentReference[oaicite:9]{index=9}
            Quaternion worldTarget = delta * bone.rotation;
            Quaternion localTarget = Quaternion.Inverse(bone.parent.rotation) * worldTarget;

            // 4️⃣ Smooth & set
            if(!prevRot.ContainsKey(kv.Value)) prevRot[kv.Value] = localTarget;
            float t = 1 - Mathf.Pow(1 - smoothFactor, Time.deltaTime * frameRate);
            Quaternion s = Quaternion.Slerp(prevRot[kv.Value], localTarget, t); // :contentReference[oaicite:10]{index=10}
            bone.localRotation = s;
            prevRot[kv.Value] = s;
        }
    }
}
