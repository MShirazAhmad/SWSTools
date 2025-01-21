# Generated by Django 5.1.5 on 2025-01-21 17:37

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('analyst_name', models.CharField(max_length=100)),
                ('file', models.FileField(upload_to='uploads/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='uploaded_files', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='GeneratedPlot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('plot', models.ImageField(upload_to='plots/')),
                ('generated_at', models.DateTimeField(auto_now_add=True)),
                ('uploaded_file', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='generated_plots', to='NanoAnalyst.uploadedfile')),
            ],
        ),
    ]
