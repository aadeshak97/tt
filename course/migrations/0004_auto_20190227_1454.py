# Generated by Django 2.1.7 on 2019-02-27 09:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('course', '0003_subject_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subject',
            name='subject_type',
            field=models.CharField(choices=[('1', 'Lab'), ('2', 'Lecture'), ('3', 'Tutorial')], max_length=3),
        ),
    ]
