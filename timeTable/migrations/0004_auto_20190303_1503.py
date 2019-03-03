# Generated by Django 2.1.7 on 2019-03-03 09:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeTable', '0003_auto_20190303_1456'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='time_table',
            name='department',
        ),
        migrations.RemoveField(
            model_name='time_table',
            name='organization',
        ),
        migrations.AddField(
            model_name='time_table',
            name='lect',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='time_table',
            name='sub',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='time_table',
            name='types',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='time_table',
            name='semester',
            field=models.IntegerField(default=0),
        ),
    ]