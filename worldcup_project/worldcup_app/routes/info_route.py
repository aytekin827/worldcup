from flask import Blueprint, render_template

from models import ranking
bp = Blueprint('info', __name__)

@bp.route('/info')
def index():
    countries = ranking.query.all()
    return render_template('info.html',countries = countries)
