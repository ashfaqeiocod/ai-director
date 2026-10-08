# clips/ — put your own test videos here

This folder is for videos you want to test with. Videos in it are **never uploaded to git**;
only this README is.

Then point `config.yaml` at them, for example:

```yaml
input:
  mode: files
  files:
    cams:
      - { source: "clips/left.mp4",   start_seconds: 0 }
      - { source: "clips/middle.mp4", start_seconds: 0 }
      - { source: "clips/right.mp4",  start_seconds: 0 }
```

Step-by-step guide: [docs/USING_YOUR_OWN_VIDEOS.md](../docs/USING_YOUR_OWN_VIDEOS.md)
