# Generated by Django 2.1.7 on 2019-02-22 13:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('organization', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Course',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('semester', models.CharField(choices=[('I', 'First'), ('II', 'Second'), ('III', 'Third'), ('IV', 'Fourth'), ('V', 'Fifth'), ('VI', 'Sixth'), ('VII', 'Seventh'), ('VIII', 'Eighth')], max_length=3)),
                ('program', models.CharField(choices=[('UG', 'Under Graduate'), ('PG', 'Post Graduate')], max_length=3)),
                ('department', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='organization.Department')),
            ],
        ),
        migrations.CreateModel(
            name='Subject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(max_length=20, unique=True)),
                ('subject_type', models.CharField(choices=[('lab', 'Lab'), ('lec', 'Lecture'), ('tut', 'Tutorial')], max_length=3)),
                ('total_duration', models.IntegerField()),
                ('period_per_week', models.IntegerField()),
                ('per_period_duration', models.IntegerField()),
                ('course', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='course.Course')),
                ('faculty', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='organization.Faculty')),
            ],
        ),
    ]
