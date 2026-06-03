import sys
from django.apps import AppConfig


class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    # Ensures the scan runs only once per process lifetime
    _scanned = False

    def ready(self):
        skip_cmds = {'migrate', 'makemigrations', 'collectstatic', 'test',
                     'shell', 'dbshell', 'createsuperuser', 'showmigrations'}
        if skip_cmds & set(sys.argv):
            return

        # Defer DB access until the first request so app init is fully complete
        from django.core.signals import request_started
        request_started.connect(self._on_first_request)

    def _on_first_request(self, sender, **kwargs):
        if SearchConfig._scanned:
            return
        SearchConfig._scanned = True

        from django.core.signals import request_started
        request_started.disconnect(self._on_first_request)

        try:
            self._scan_and_enqueue()
        except Exception:
            pass

    def _scan_and_enqueue(self):
        from pathlib import Path
        from django.conf import settings
        from .models import Video, IndexingJob
        from .tasks import index_video

        videos_dir = settings.VIDEOS_DIR
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

        for video_file in sorted(videos_dir.iterdir()):
            if video_file.suffix.lower() not in video_extensions:
                continue

            video, created = Video.objects.get_or_create(
                filename=video_file.name,
                defaults={'status': 'pending'},
            )

            if video.status in ('processing', 'indexed'):
                continue

            job, _ = IndexingJob.objects.get_or_create(video=video)
            task = index_video.delay(video.id)

            job.celery_task_id = task.id
            job.status = 'queued'
            job.save(update_fields=['celery_task_id', 'status'])

            video.status = 'processing'
            video.save(update_fields=['status'])
