"""
Manually trigger indexing of all unprocessed videos in data/videos/.
Useful for re-running after a crash or for testing outside of runserver.

    python manage.py index_videos
    python manage.py index_videos --force    # re-index even already-indexed videos
"""
from django.core.management.base import BaseCommand
from django.conf import settings

from search.models import Video, IndexingJob
from search.tasks import index_video


class Command(BaseCommand):
    help = 'Index all unprocessed videos in data/videos/'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='Re-index already-indexed videos')

    def handle(self, *args, **options):
        force = options['force']
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        videos_dir = settings.VIDEOS_DIR
        queued = 0

        for video_file in sorted(videos_dir.iterdir()):
            if video_file.suffix.lower() not in video_extensions:
                continue

            video, created = Video.objects.get_or_create(
                filename=video_file.name,
                defaults={'status': 'pending'},
            )

            if not force and video.status in ('processing', 'indexed'):
                self.stdout.write(f'  skip  {video_file.name} ({video.status})')
                continue

            if force:
                video.status = 'pending'
                video.save(update_fields=['status'])

            job, _ = IndexingJob.objects.get_or_create(video=video)
            task = index_video.delay(video.id)

            job.celery_task_id = task.id
            job.status = 'queued'
            job.progress_pct = 0
            job.current_step = ''
            job.error_msg = ''
            job.save()

            video.status = 'processing'
            video.save(update_fields=['status'])

            self.stdout.write(self.style.SUCCESS(f'  queued {video_file.name}'))
            queued += 1

        self.stdout.write(self.style.SUCCESS(f'\n{queued} video(s) queued for indexing.'))
