# Generated by Django 2.1.7 on 2019-03-03 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeTable', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='time_table',
            name='days',
        ),
        migrations.AddField(
            model_name='time_table',
            name='day',
            field=models.IntegerField(default=0),
        ),
    ]
