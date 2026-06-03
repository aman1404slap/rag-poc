from django.db import models


class Video(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('indexed', 'Indexed'),
        ('error', 'Error'),
    ]

    filename = models.CharField(max_length=255, unique=True)
    duration_sec = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    indexed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename

    @property
    def video_url(self):
        return f'/media/videos/{self.filename}'


class Clip(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='clips')
    clip_index = models.IntegerField()
    start_sec = models.FloatField()
    end_sec = models.FloatField()
    caption = models.TextField(blank=True)
    ocr_text = models.TextField(blank=True)
    action_label = models.CharField(max_length=150, blank=True)
    action_confidence = models.FloatField(null=True, blank=True)
    actions_detected = models.JSONField(default=list)  # [{label, confidence}, ...]
    objects_detected = models.JSONField(default=list)  # [{label, confidence}, ...]
    ocr_blocks = models.JSONField(default=list)  # [{text, confidence}, ...]
    caption_confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['video', 'clip_index']

    def __str__(self):
        return f'{self.video.filename} [{self.start_sec:.1f}s–{self.end_sec:.1f}s]'

    @property
    def chroma_id(self):
        return str(self.id)


class Keyframe(models.Model):
    clip = models.ForeignKey(Clip, on_delete=models.CASCADE, related_name='keyframes')
    # path relative to MEDIA_ROOT (data/)
    file_path = models.CharField(max_length=500)
    timestamp_sec = models.FloatField()

    @property
    def url(self):
        return f'/media/{self.file_path}'


class IndexingJob(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('done', 'Done'),
        ('error', 'Error'),
    ]

    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='job')
    celery_task_id = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    progress_pct = models.IntegerField(default=0)
    current_step = models.CharField(max_length=200, blank=True)
    error_msg = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'Job({self.video.filename}, {self.status}, {self.progress_pct}%)'
