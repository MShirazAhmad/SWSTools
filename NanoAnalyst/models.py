# uploader/models.py

from django.db import models

# uploader/models.py

from django.db import models
from django.contrib.auth.models import User

class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file.name} Uploaded by {self.user.username}"

class GeneratedPlot(models.Model):
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='generated_plots')
    plot = models.ImageField(upload_to='plots/')
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Plot generated from {self.uploaded_file.file.name} at {self.generated_at}"
