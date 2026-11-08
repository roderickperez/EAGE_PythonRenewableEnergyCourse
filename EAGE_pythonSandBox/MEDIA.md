# Course audio and video

Put your own participant-safe media in the following folders:

- `media/en/` — English section audio and lesson videos.
- `media/es/` — Spanish section audio and lesson videos.

Add each file to `media.json`, using the exact lesson ID and section heading ID
from `generated/course.json`. Example (replace the IDs and filenames with real ones):

```json
{
  "lesson-id.md": {
    "audio": {
      "en": {"intro": "en/lesson-intro.mp3", "section-id": "en/lesson-section.mp3"},
      "es": {"intro": "es/lesson-intro.mp3", "section-id": "es/lesson-section.mp3"}
    },
    "video": {"en": "en/lesson.mp4", "es": "es/lesson.mp4"}
  }
}
```

A section microphone plays its recording. The lesson audio button plays the
optional intro followed by all section recordings in page order; it is enabled
only when every section has a recording. The lesson video button plays its
language's video. Missing recordings remain disabled. The builder validates
lesson IDs, heading IDs, file types and paths, and copies the mapped files into
the student course. Do not place private or instructor-only files here.
