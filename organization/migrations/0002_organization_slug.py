# Generated by Django 2.1.7 on 2019-02-23 15:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('organization', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='organization',
            name='slug',
            field=models.SlugField(default=0),
            preserve_default=False,
        ),
    ]
