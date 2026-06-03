"""
Wipe all extracted data and vector index, reset videos to pending.

    python manage.py reset_index
"""
import shutil
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Delete all clips, keyframes, jobs, ChromaDB index; reset videos to pending.'

    def handle(self, *args, **options):
        from search.models import Video, Clip, Keyframe, IndexingJob

        # DB
        n_jobs = IndexingJob.objects.all().delete()[0]
        n_kf   = Keyframe.objects.all().delete()[0]
        n_cl   = Clip.objects.all().delete()[0]
        Video.objects.all().update(status='pending', indexed_at=None, duration_sec=None)
        self.stdout.write(f'  DB: deleted {n_cl} clips, {n_kf} keyframes, {n_jobs} jobs; videos reset to pending')

        # Keyframe images
        kf_dir = settings.KEYFRAMES_DIR
        if kf_dir.exists():
            shutil.rmtree(kf_dir)
            kf_dir.mkdir()
        self.stdout.write(f'  Keyframes dir wiped: {kf_dir}')

        # ChromaDB index
        idx_dir = settings.INDEX_DIR
        if idx_dir.exists():
            shutil.rmtree(idx_dir)
            idx_dir.mkdir()
        self.stdout.write(f'  ChromaDB index wiped: {idx_dir}')

        self.stdout.write(self.style.SUCCESS('Done. Run `python manage.py index_videos` or restart the server to re-index.'))
