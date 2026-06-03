from django.contrib import admin
from .models import Video, Clip, Keyframe, IndexingJob


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('filename', 'status', 'duration_sec', 'indexed_at', 'created_at')
    list_filter = ('status',)


@admin.register(Clip)
class ClipAdmin(admin.ModelAdmin):
    list_display = ('video', 'clip_index', 'start_sec', 'end_sec', 'action_label')
    list_filter = ('video',)


@admin.register(IndexingJob)
class IndexingJobAdmin(admin.ModelAdmin):
    list_display = ('video', 'status', 'progress_pct', 'current_step', 'created_at')
    list_filter = ('status',)
