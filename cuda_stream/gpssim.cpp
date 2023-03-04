#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#include "kernal.h"

#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>
#endif
#include "gpssim.h"
//TPoint *point;
int countPoint = 0;

int sinTable512[] = {
	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
};



int cosTable512[] = {
	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
};

// Receiver antenna attenuation in dB for boresight angle = 0:5:180 [deg]//boresight 视轴是天线增益最大的地方，所以零度的时候为0，此时增益最高
double ant_pat_db[37] = {
	0.00,  0.00,  0.22,  0.44,  0.67,  1.11,  1.56,  2.00,  2.44,  2.89,  3.56,  4.22,
	4.89,  5.56,  6.22,  6.89,  7.56,  8.22,  8.89,  9.78, 10.67, 11.56, 12.44, 13.33,
	14.44, 15.56, 16.67, 17.78, 18.89, 20.00, 21.33, 22.67, 24.00, 25.56, 27.33, 29.33,
	31.56
};



/*! \brief Subtract two vectors of double
*  \param[out] y Result of subtraction
*  \param[in] x1 Minuend of subtracion
*  \param[in] x2 Subtrahend of subtracion
*/
void subVect(double *y, const double *x1, const double *x2)
{
	y[0] = x1[0] - x2[0];
	y[1] = x1[1] - x2[1];
	y[2] = x1[2] - x2[2];

	return;
}
/// <mgrn-生成服从高斯分布的随机数>
/// 
/// </summary>
/// <param name="u"></param>
/// <param name="g"></param>
/// <param name="r"></param>
/// <returns></随机噪声>
double mgrn1(double u, double g, double* r)
{
	int i, m;
	double s, w, v, t;
	s = 65536.0; w = 2053.0; v = 13849.0;
	t = 0.0;
	for (i = 1; i <= 12; i++)
	{
		*r = (*r) * w + v; m = (int)(*r / s);
		*r = *r - m * s; t = t + (*r) / s;
	}
	t = u + g * (t - 6.0);
	return(t);
}

//double mpow(double base, double index)
//{
//	double temp = base;
//	for (int i = 0; i < index-1; i++)
//	{
//		base = base * temp;
//	}
//	return base;
//}
/*! \brief Compute Norm of Vector
*  \param[in] x Input vector
*  \returns Length (Norm) of the input vector
*/
double normVect(const double *x)
{
	return(sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
}

/*! \brief Compute dot-product of two vectors
*  \param[in] x1 First multiplicand
*  \param[in] x2 Second multiplicand
*  \returns Dot-product of both multiplicands
*/
double dotProd(const double *x1, const double *x2)
{
	return(x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2]);
}

/* !\brief generate the C/A code sequence for a given Satellite Vehicle PRN
*  \param[in] prn PRN nuber of the Satellite Vehicle
*  \param[out] ca Caller-allocated integer array of 1023 bytes
*/
// sinh C/A code
void codegen(int* p, int num)
{
	int gs2[] = { 5,6,7,8,17,18,139,140,141,251,252,254,255,256,257,258,469,470,471,\
		472,473,474,509,512,513,514,515,516,859,860,861,862 };
	int save1 = 0, save2 = 0;
	int g2shift = gs2[num - 1];
	int g1[1023] = { 0 };
	int g2[1023] = { 0 };
	int g2_temp[1023] = { 0 };
	int reg[10] = { 0 };
	for (int i = 0; i < 10; i++)
	{
		reg[i] = -1;
	}
	for (int i = 0; i < 1023; i++)
	{
		g1[i] = reg[9];
		save1 = reg[2] * reg[9];
		for (int j = 9; j > 0; j--)
		{
			reg[j] = reg[j - 1];
		}
		reg[0] = save1;
	}
	for (int i = 0; i < 10; i++)
	{
		reg[i] = -1;
	}
	for (int i = 0; i < 1023; i++)
	{
		g2[i] = reg[9];
		save2 = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9];
		for (int j = 9; j > 0; j--)
		{
			reg[j] = reg[j - 1];
		}
		reg[0] = save2;
	}
	for (int i = 0; i < g2shift; i++)
	{
		g2_temp[i] = g2[1023 - g2shift + i];
	}
	for (int i = g2shift; i < 1023; i++)
	{
		g2_temp[i] = g2[i - g2shift];
	}
	for (int i = 0; i < 1023; i++)
	{
		p[i] = -(g1[i] * g2_temp[i]);
	}
}

/*! \brief Convert a UTC date into a GPS date
*  \param[in] t input date in UTC form
*  \param[out] g output date in GPS form
*/
void date2gps(const datetime_t *t, gpstime_t *g)
{
	int doy[12] = { 0,31,59,90,120,151,181,212,243,273,304,334 };//每月第一天对应年中的天数
	int ye;
	int de;
	int lpdays;

	ye = t->y - 1980;

	// Compute the number of leap days since Jan 5/Jan 6, 1980.
	lpdays = ye / 4 + 1;
	if ((ye % 4) == 0 && t->m <= 2)
		lpdays--;

	// Compute the number of days elapsed since Jan 5/Jan 6, 1980.
	de = ye * 365 + doy[t->m - 1] + t->d + lpdays - 6;

	// Convert time to GPS weeks and seconds.
	g->week = de / 7;
	g->sec = (double)(de % 7)*SECONDS_IN_DAY + t->hh*SECONDS_IN_HOUR
		+ t->mm*SECONDS_IN_MINUTE + t->sec;

	return;
}

void gps2date(const gpstime_t *g, datetime_t *t)
{
	// Convert Julian day number to calendar date
	int c = (int)(7 * g->week + floor(g->sec / 86400.0) + 2444245.0) + 1537;
	int d = (int)((c - 122.1) / 365.25);
	int e = 365 * d + d / 4;
	int f = (int)((c - e) / 30.6001);

	t->d = c - e - (int)(30.6001*f);
	t->m = f - 1 - 12 * (f / 14);
	t->y = d - 4715 - ((7 + t->m) / 10);

	t->hh = ((int)(g->sec / 3600.0)) % 24;
	t->mm = ((int)(g->sec / 60.0)) % 60;
	t->sec = g->sec - 60.0*floor(g->sec / 60.0);

	return;
}

/*! \brief Convert Earth-centered Earth-fixed (ECEF) into Lat/Long/Heighth
*  \param[in] xyz Input Array of X, Y and Z ECEF coordinates
*  \param[out] llh Output Array of Latitude, Longitude and Height
*/
void xyz2llh(const double *xyz, double *llh)
{
	double a, eps, e, e2;
	double x, y, z;
	double rho2, dz, zdz, nh, slat, n, dz_new;

	a = WGS84_RADIUS;
	e = WGS84_ECCENTRICITY;

	eps = 1.0e-3;
	e2 = e*e;

	if (normVect(xyz) < eps)
	{
		// Invalid ECEF vector
		llh[0] = 0.0;
		llh[1] = 0.0;
		llh[2] = -a;

		return;
	}

	x = xyz[0];
	y = xyz[1];
	z = xyz[2];

	rho2 = x*x + y*y;
	dz = e2*z;

	while (1)
	{
		zdz = z + dz;
		nh = sqrt(rho2 + zdz*zdz);
		slat = zdz / nh;
		n = a / sqrt(1.0 - e2*slat*slat);
		dz_new = n*e2*slat;

		if (fabs(dz - dz_new) < eps)
			break;

		dz = dz_new;
	}

	llh[0] = atan2(zdz, sqrt(rho2));
	llh[1] = atan2(y, x);
	llh[2] = nh - n;

	return;
}

/*! \brief Convert Lat/Long/Height into Earth-centered Earth-fixed (ECEF)
*  \param[in] llh Input Array of Latitude, Longitude and Height
*  \param[out] xyz Output Array of X, Y and Z ECEF coordinates
*/
void llh2xyz(const double *llh, double *xyz)
{
	double n;
	double a;
	double e;
	double e2;
	double clat;
	double slat;
	double clon;
	double slon;
	double d, nph;
	double tmp;

	a = WGS84_RADIUS;
	e = WGS84_ECCENTRICITY;
	e2 = e*e;

	clat = cos(llh[0]);
	slat = sin(llh[0]);
	clon = cos(llh[1]);
	slon = sin(llh[1]);
	d = e*slat;

	n = a / sqrt(1.0 - d*d);
	nph = n + llh[2];

	tmp = nph*clat;
	xyz[0] = tmp*clon;
	xyz[1] = tmp*slon;
	xyz[2] = ((1.0 - e2)*n + llh[2])*slat;

	return;
}

/*! \brief Compute the intermediate matrix for LLH to ECEF
*  \param[in] llh Input position in Latitude-Longitude-Height format
*  \param[out] t Three-by-Three output matrix
*/
void ltcmat(const double *llh, double t[3][3])
{
	double slat, clat;
	double slon, clon;

	slat = sin(llh[0]);
	clat = cos(llh[0]);
	slon = sin(llh[1]);
	clon = cos(llh[1]);

	t[0][0] = -slat*clon;
	t[0][1] = -slat*slon;
	t[0][2] = clat;
	t[1][0] = -slon;
	t[1][1] = clon;
	t[1][2] = 0.0;
	t[2][0] = clat*clon;
	t[2][1] = clat*slon;
	t[2][2] = slat;

	return;
}

/*! \brief Convert Earth-centered Earth-Fixed to ?
*  \param[in] xyz Input position as vector in ECEF format
*  \param[in] t Intermediate matrix computed by \ref ltcmat
*  \param[out] neu Output position as North-East-Up format
*/
void ecef2neu(const double *xyz, double t[3][3], double *neu)
{
	neu[0] = t[0][0] * xyz[0] + t[0][1] * xyz[1] + t[0][2] * xyz[2];
	neu[1] = t[1][0] * xyz[0] + t[1][1] * xyz[1] + t[1][2] * xyz[2];
	neu[2] = t[2][0] * xyz[0] + t[2][1] * xyz[1] + t[2][2] * xyz[2];

	return;
}

/*! \brief Convert North-Eeast-Up to Azimuth + Elevation
*  \param[in] neu Input position in North-East-Up format
*  \param[out] azel Output array of azimuth + elevation as double
*/
void neu2azel(double *azel, const double *neu)
{
	double ne;

	azel[0] = atan2(neu[1], neu[0]);
	if (azel[0] < 0.0)
		azel[0] += (2.0*PI);

	ne = sqrt(neu[0] * neu[0] + neu[1] * neu[1]);
	azel[1] = atan2(neu[2], ne);

	return;
}

/*! \brief Compute Satellite position, velocity and clock at given time
*  \param[in] eph Ephemeris data of the satellite
*  \param[in] g GPS time at which position is to be computed
*  \param[out] pos Computed position (vector)
*  \param[out] vel Computed velociy (vector)
*  \param[clk] clk Computed clock
*/
void satpos(ephem_t eph, gpstime_t g, double *pos, double *vel, double *clk)
{
	// Computing Satellite Velocity using the Broadcast Ephemeris
	// http://www.ngs.noaa.gov/gps-toolbox/bc_velo.htm

	double tk;
	double mk;
	double ek;
	double ekold;
	double ekdot;
	double cek, sek;
	double pk;
	double pkdot;
	double c2pk, s2pk;
	double uk;
	double ukdot;
	double cuk, suk;
	double ok;
	double sok, cok;
	double ik;
	double ikdot;
	double sik, cik;
	double rk;
	double rkdot;
	double xpk, ypk;
	double xpkdot, ypkdot;

	double relativistic, OneMinusecosE, tmp;

	tk = g.sec - eph.toe.sec;//

	if (tk > SECONDS_IN_HALF_WEEK)//
		tk -= SECONDS_IN_WEEK;
	else if (tk < -SECONDS_IN_HALF_WEEK)//
		tk += SECONDS_IN_WEEK;

	mk = eph.m0 + eph.n*tk;
	ek = mk;
	ekold = ek + 1.0;

	OneMinusecosE = 0; // Suppress the uninitialized warning.
	while (fabs(ek - ekold) > 1.0E-14)
	{
		ekold = ek;
		OneMinusecosE = 1.0 - eph.ecc*cos(ekold);
		ek = ek + (mk - ekold + eph.ecc*sin(ekold)) / OneMinusecosE;
	}

	sek = sin(ek);
	cek = cos(ek);

	ekdot = eph.n / OneMinusecosE;

	relativistic = -4.442807633E-10*eph.ecc*eph.sqrta*sek;

	pk = atan2(eph.sq1e2*sek, cek - eph.ecc) + eph.aop;
	pkdot = eph.sq1e2*ekdot / OneMinusecosE;

	s2pk = sin(2.0*pk);
	c2pk = cos(2.0*pk);

	uk = pk + eph.cus*s2pk + eph.cuc*c2pk;
	suk = sin(uk);
	cuk = cos(uk);
	ukdot = pkdot*(1.0 + 2.0*(eph.cus*c2pk - eph.cuc*s2pk));

	rk = eph.A*OneMinusecosE + eph.crc*c2pk + eph.crs*s2pk;
	rkdot = eph.A*eph.ecc*sek*ekdot + 2.0*pkdot*(eph.crs*c2pk - eph.crc*s2pk);

	ik = eph.inc0 + eph.idot*tk + eph.cic*c2pk + eph.cis*s2pk;
	sik = sin(ik);
	cik = cos(ik);
	ikdot = eph.idot + 2.0*pkdot*(eph.cis*c2pk - eph.cic*s2pk);

	xpk = rk*cuk;
	ypk = rk*suk;
	xpkdot = rkdot*cuk - ypk*ukdot;
	ypkdot = rkdot*suk + xpk*ukdot;

	ok = eph.omg0 + tk*eph.omgkdot - OMEGA_EARTH*eph.toe.sec;//和书上的公式不一样，这样计算出来的是在ecef坐标系下吗？
	sok = sin(ok);
	cok = cos(ok);

	pos[0] = xpk*cok - ypk*cik*sok;
	pos[1] = xpk*sok + ypk*cik*cok;
	pos[2] = ypk*sik;

	tmp = ypkdot*cik - ypk*sik*ikdot;

	vel[0] = -eph.omgkdot*pos[1] + xpkdot*cok - tmp*sok;
	vel[1] = eph.omgkdot*pos[0] + xpkdot*sok + tmp*cok;
	vel[2] = ypk*cik*ikdot + ypkdot*sik;

	// Satellite clock correction
	tk = g.sec - eph.toc.sec;

	if (tk > SECONDS_IN_HALF_WEEK)
		tk -= SECONDS_IN_WEEK;
	else if (tk < -SECONDS_IN_HALF_WEEK)
		tk += SECONDS_IN_WEEK;

	clk[0] = eph.af0 + tk*(eph.af1 + tk*eph.af2) + relativistic - eph.tgd;
	clk[1] = eph.af1 + 2.0*tk*eph.af2;

	return;
}
/// <summary>
/// 
/// </summary>
/// <param name="src"></param>
/// <param name="mode"></param>
/// <returns></returns>
unsigned char quantify(float src, int mode)//mode用于选择量化为xbit，1bit/2bit
{
	if (mode == 2)
	{
		if (src > quantify_th)
		{
			return 0x01;//3
		}
		else if (src < quantify_th && src>0)
		{
			return 0x00;//1
		}
		else if (src < -quantify_th)
		{
			return 0x03;//-3
		}
		else if (src > -quantify_th && src < 0)
		{
			return 0x02;//-1
		}
	}
	if (mode == 1)
	{
		if (src > 0)
			return 0x01;
		else if (src < 0)
			return 0x00;
	}
	else
	{
		printf("error:(quantify) unexpected mode");
		return -10;
	}
}
/*! \brief Compute Subframe from Ephemeris
*  \param[in] eph Ephemeris of given SV
*  \param[out] sbf Array of five sub-frames, 10 long words each
*/
void eph2sbf(const ephem_t eph, const ionoutc_t ionoutc, unsigned long sbf[5][N_DWRD_SBF])
{
	unsigned long wn;
	unsigned long toe;
	unsigned long toc;
	unsigned long iode;
	unsigned long iodc;
	long deltan;
	long cuc;
	long cus;
	long cic;
	long cis;
	long crc;
	long crs;
	unsigned long ecc;
	unsigned long sqrta;
	long m0;
	long omg0;
	long inc0;
	long aop;
	long omgdot;
	long idot;
	long af0;
	long af1;
	long af2;
	long tgd;

	unsigned long ura = 2UL;
	unsigned long dataId = 1UL;
	unsigned long sbf4_page25_svId = 63UL;
	unsigned long sbf5_page25_svId = 51UL;

	unsigned long wna;
	unsigned long toa;

	signed long alpha0, alpha1, alpha2, alpha3;
	signed long beta0, beta1, beta2, beta3;
	signed long A0, A1;
	signed long dtls, dtlsf;
	unsigned long tot, wnt, wnlsf, dn;
	unsigned long sbf4_page18_svId = 56UL;

	// FIXED: This has to be the "transmission" week number, not for the ephemeris reference time
	//wn = (unsigned long)(eph.toe.week%1024);
	wn = 0UL;
	toe = (unsigned long)(eph.toe.sec / 16.0);
	toc = (unsigned long)(eph.toc.sec / 16.0);
	iode = (unsigned long)(eph.iode);
	iodc = (unsigned long)(eph.iodc);
	deltan = (long)(eph.deltan / POW2_M43 / PI);
	cuc = (long)(eph.cuc / POW2_M29);
	cus = (long)(eph.cus / POW2_M29);
	cic = (long)(eph.cic / POW2_M29);
	cis = (long)(eph.cis / POW2_M29);
	crc = (long)(eph.crc / POW2_M5);
	crs = (long)(eph.crs / POW2_M5);
	ecc = (unsigned long)(eph.ecc / POW2_M33);
	sqrta = (unsigned long)(eph.sqrta / POW2_M19);
	m0 = (long)(eph.m0 / POW2_M31 / PI);
	omg0 = (long)(eph.omg0 / POW2_M31 / PI);
	inc0 = (long)(eph.inc0 / POW2_M31 / PI);
	aop = (long)(eph.aop / POW2_M31 / PI);
	omgdot = (long)(eph.omgdot / POW2_M43 / PI);
	idot = (long)(eph.idot / POW2_M43 / PI);
	af0 = (long)(eph.af0 / POW2_M31);
	af1 = (long)(eph.af1 / POW2_M43);
	af2 = (long)(eph.af2 / POW2_M55);
	tgd = (long)(eph.tgd / POW2_M31);

	wna = (unsigned long)(eph.toe.week % 256);
	toa = (unsigned long)(eph.toe.sec / 4096.0);

	alpha0 = (signed long)round(ionoutc.alpha0 / POW2_M30);
	alpha1 = (signed long)round(ionoutc.alpha1 / POW2_M27);
	alpha2 = (signed long)round(ionoutc.alpha2 / POW2_M24);
	alpha3 = (signed long)round(ionoutc.alpha3 / POW2_M24);
	beta0 = (signed long)round(ionoutc.beta0 / 2048.0);
	beta1 = (signed long)round(ionoutc.beta1 / 16384.0);
	beta2 = (signed long)round(ionoutc.beta2 / 65536.0);
	beta3 = (signed long)round(ionoutc.beta3 / 65536.0);
	A0 = (signed long)round(ionoutc.A0 / POW2_M30);
	A1 = (signed long)round(ionoutc.A1 / POW2_M50);
	dtls = (signed long)(ionoutc.dtls);
	tot = (unsigned long)(ionoutc.tot / 4096);
	wnt = (unsigned long)(ionoutc.wnt % 256);
	// TO DO: Specify scheduled leap seconds in command options
	// 2016/12/31 (Sat) -> WNlsf = 1929, DN = 7 (http://navigationservices.agi.com/GNSSWeb/)
	// Days are counted from 1 to 7 (Sunday is 1).
	wnlsf = 1929 % 256;
	dn = 7;
	dtlsf = 18;

	// Subframe 1
	sbf[0][0] = 0x8B0000UL << 6;
	sbf[0][1] = 0x1UL << 8;
	sbf[0][2] = ((wn & 0x3FFUL) << 20) | (ura << 14) | (((iodc >> 8) & 0x3UL) << 6);
	sbf[0][3] = 0UL;
	sbf[0][4] = 0UL;
	sbf[0][5] = 0UL;
	sbf[0][6] = (tgd & 0xFFUL) << 6;
	sbf[0][7] = ((iodc & 0xFFUL) << 22) | ((toc & 0xFFFFUL) << 6);
	sbf[0][8] = ((af2 & 0xFFUL) << 22) | ((af1 & 0xFFFFUL) << 6);
	sbf[0][9] = (af0 & 0x3FFFFFUL) << 8;

	// Subframe 2
	sbf[1][0] = 0x8B0000UL << 6;
	sbf[1][1] = 0x2UL << 8;
	sbf[1][2] = ((iode & 0xFFUL) << 22) | ((crs & 0xFFFFUL) << 6);
	sbf[1][3] = ((deltan & 0xFFFFUL) << 14) | (((m0 >> 24) & 0xFFUL) << 6);
	sbf[1][4] = (m0 & 0xFFFFFFUL) << 6;
	sbf[1][5] = ((cuc & 0xFFFFUL) << 14) | (((ecc >> 24) & 0xFFUL) << 6);
	sbf[1][6] = (ecc & 0xFFFFFFUL) << 6;
	sbf[1][7] = ((cus & 0xFFFFUL) << 14) | (((sqrta >> 24) & 0xFFUL) << 6);
	sbf[1][8] = (sqrta & 0xFFFFFFUL) << 6;
	sbf[1][9] = (toe & 0xFFFFUL) << 14;

	// Subframe 3
	sbf[2][0] = 0x8B0000UL << 6;
	sbf[2][1] = 0x3UL << 8;
	sbf[2][2] = ((cic & 0xFFFFUL) << 14) | (((omg0 >> 24) & 0xFFUL) << 6);
	sbf[2][3] = (omg0 & 0xFFFFFFUL) << 6;
	sbf[2][4] = ((cis & 0xFFFFUL) << 14) | (((inc0 >> 24) & 0xFFUL) << 6);
	sbf[2][5] = (inc0 & 0xFFFFFFUL) << 6;
	sbf[2][6] = ((crc & 0xFFFFUL) << 14) | (((aop >> 24) & 0xFFUL) << 6);
	sbf[2][7] = (aop & 0xFFFFFFUL) << 6;
	sbf[2][8] = (omgdot & 0xFFFFFFUL) << 6;
	sbf[2][9] = ((iode & 0xFFUL) << 22) | ((idot & 0x3FFFUL) << 8);

	if (ionoutc.vflg == TRUE)
	{
		// Subframe 4, page 18
		sbf[3][0] = 0x8B0000UL << 6;
		sbf[3][1] = 0x4UL << 8;
		sbf[3][2] = (dataId << 28) | (sbf4_page18_svId << 22) | ((alpha0 & 0xFFUL) << 14) | ((alpha1 & 0xFFUL) << 6);
		sbf[3][3] = ((alpha2 & 0xFFUL) << 22) | ((alpha3 & 0xFFUL) << 14) | ((beta0 & 0xFFUL) << 6);
		sbf[3][4] = ((beta1 & 0xFFUL) << 22) | ((beta2 & 0xFFUL) << 14) | ((beta3 & 0xFFUL) << 6);
		sbf[3][5] = (A1 & 0xFFFFFFUL) << 6;
		sbf[3][6] = ((A0 >> 8) & 0xFFFFFFUL) << 6;
		sbf[3][7] = ((A0 & 0xFFUL) << 22) | ((tot & 0xFFUL) << 14) | ((wnt & 0xFFUL) << 6);
		sbf[3][8] = ((dtls & 0xFFUL) << 22) | ((wnlsf & 0xFFUL) << 14) | ((dn & 0xFFUL) << 6);
		sbf[3][9] = (dtlsf & 0xFFUL) << 22;

	}
	else
	{
		// Subframe 4, page 25
		sbf[3][0] = 0x8B0000UL << 6;
		sbf[3][1] = 0x4UL << 8;
		sbf[3][2] = (dataId << 28) | (sbf4_page25_svId << 22);
		sbf[3][3] = 0UL;
		sbf[3][4] = 0UL;
		sbf[3][5] = 0UL;
		sbf[3][6] = 0UL;
		sbf[3][7] = 0UL;
		sbf[3][8] = 0UL;
		sbf[3][9] = 0UL;
	}

	// Subframe 5, page 25
	sbf[4][0] = 0x8B0000UL << 6;
	sbf[4][1] = 0x5UL << 8;
	sbf[4][2] = (dataId << 28) | (sbf5_page25_svId << 22) | ((toa & 0xFFUL) << 14) | ((wna & 0xFFUL) << 6);
	sbf[4][3] = 0UL;
	sbf[4][4] = 0UL;
	sbf[4][5] = 0UL;
	sbf[4][6] = 0UL;
	sbf[4][7] = 0UL;
	sbf[4][8] = 0UL;
	sbf[4][9] = 0UL;

	return;
}

/*! \brief Count number of bits set to 1
*  \param[in] v long word in whihc bits are counted
*  \returns Count of bits set to 1
*/
unsigned long countBits(unsigned long v)
{
	unsigned long c;
	const int S[] = { 1, 2, 4, 8, 16 };
	const unsigned long B[] = {
		0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF };

	c = v;
	c = ((c >> S[0]) & B[0]) + (c & B[0]);
	c = ((c >> S[1]) & B[1]) + (c & B[1]);
	c = ((c >> S[2]) & B[2]) + (c & B[2]);
	c = ((c >> S[3]) & B[3]) + (c & B[3]);
	c = ((c >> S[4]) & B[4]) + (c & B[4]);

	return(c);
}

/*! \brief Compute the Checksum for one given word of a subframe
*  \param[in] source The input data
*  \param[in] nib Does this word contain non-information-bearing bits?
*  \returns Computed Checksum
*/
unsigned long computeChecksum(unsigned long source, int nib)
{
	/*
	Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
	Bits 29 to  6 = Source data bits, d1, d2, ..., d24
	Bits  5 to  0 = Empty parity bits
	*/

	/*
	Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
	Bits 29 to  6 = Data bits transmitted by the SV, D1, D2, ..., D24
	Bits  5 to  0 = Computed parity bits, D25, D26, ..., D30
	*/

	/*
	1            2           3
	bit    12 3456 7890 1234 5678 9012 3456 7890
	---    -------------------------------------
	D25    11 1011 0001 1111 0011 0100 1000 0000
	D26    01 1101 1000 1111 1001 1010 0100 0000
	D27    10 1110 1100 0111 1100 1101 0000 0000
	D28    01 0111 0110 0011 1110 0110 1000 0000
	D29    10 1011 1011 0001 1111 0011 0100 0000
	D30    00 1011 0111 1010 1000 1001 1100 0000
	*/

	unsigned long bmask[6] = {
		0x3B1F3480UL, 0x1D8F9A40UL, 0x2EC7CD00UL,
		0x1763E680UL, 0x2BB1F340UL, 0x0B7A89C0UL };

	unsigned long D;
	unsigned long d = source & 0x3FFFFFC0UL;
	unsigned long D29 = (source >> 31) & 0x1UL;
	unsigned long D30 = (source >> 30) & 0x1UL;

	if (nib) // Non-information bearing bits for word 2 and 10
	{
		/*
		Solve bits 23 and 24 to presearve parity check
		with zeros in bits 29 and 30.
		*/

		if ((D30 + countBits(bmask[4] & d)) % 2)
			d ^= (0x1UL << 6);
		if ((D29 + countBits(bmask[5] & d)) % 2)
			d ^= (0x1UL << 7);
	}

	D = d;
	if (D30)
		D ^= 0x3FFFFFC0UL;

	D |= ((D29 + countBits(bmask[0] & d)) % 2) << 5;
	D |= ((D30 + countBits(bmask[1] & d)) % 2) << 4;
	D |= ((D29 + countBits(bmask[2] & d)) % 2) << 3;
	D |= ((D30 + countBits(bmask[3] & d)) % 2) << 2;
	D |= ((D30 + countBits(bmask[4] & d)) % 2) << 1;
	D |= ((D29 + countBits(bmask[5] & d)) % 2);

	D &= 0x3FFFFFFFUL;
	//D |= (source & 0xC0000000UL); // Add D29* and D30* from source data bits

	return(D);
}

/*! \brief Replace all 'E' exponential designators to 'D'
*  \param str String in which all occurrences of 'E' are replaced with *  'D'
*  \param len Length of input string in bytes
*  \returns Number of characters replaced
*/
int replaceExpDesignator(char *str, int len)
{
	int i, n = 0;

	for (i = 0; i < len; i++)
	{
		if (str[i] == 'D')
		{
			n++;
			str[i] = 'E';
		}
	}

	return(n);
}

double subGpsTime(gpstime_t g1, gpstime_t g0)
{
	double dt;

	dt = g1.sec - g0.sec;
	dt += (double)(g1.week - g0.week) * SECONDS_IN_WEEK;

	return(dt);
}

gpstime_t incGpsTime(gpstime_t g0, double dt)
{
	gpstime_t g1;

	g1.week = g0.week;
	g1.sec = g0.sec + dt;

	g1.sec = round(g1.sec*1000.0) / 1000.0; // Avoid rounding error

	while (g1.sec >= SECONDS_IN_WEEK)
	{
		g1.sec -= SECONDS_IN_WEEK;
		g1.week++;
	}

	while (g1.sec < 0.0)
	{
		g1.sec += SECONDS_IN_WEEK;
		g1.week--;
	}

	return(g1);
}

/*! \brief Read Ephemersi data from the RINEX Navigation file */
/*  \param[out] eph Array of Output SV ephemeris data
*  \param[in] fname File name of the RINEX file
*  \returns Number of sets of ephemerides in the file
*/
int readRinexNavAll(ephem_t eph[][MAX_SAT], ionoutc_t *ionoutc, const char *fname)//读取的是fname阶段
{
	FILE *fp;
	int ieph;

	int sv;
	char str[MAX_CHAR];
	char tmp[20];

	datetime_t t;
	gpstime_t g;
	gpstime_t g0;
	double dt;

	int flags = 0x0;

	if (NULL == (fp = fopen(fname, "rt")))
		return(-1);

	// Clear valid flag
	for (ieph = 0; ieph < EPHEM_ARRAY_SIZE; ieph++)
		for (sv = 0; sv < MAX_SAT; sv++)
			eph[ieph][sv].vflg = 0;//清除可用信号的标志位

	// Read header lines
	while (1)
	{
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		if (strncmp(str + 60, "END OF HEADER", 13) == 0)
			break;
		else if (strncmp(str + 60, "ION ALPHA", 9) == 0)
		{
			strncpy(tmp, str + 2, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->alpha0 = atof(tmp);

			strncpy(tmp, str + 14, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->alpha1 = atof(tmp);

			strncpy(tmp, str + 26, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->alpha2 = atof(tmp);

			strncpy(tmp, str + 38, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->alpha3 = atof(tmp);

			flags |= 0x1;
		}
		else if (strncmp(str + 60, "ION BETA", 8) == 0)
		{
			strncpy(tmp, str + 2, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->beta0 = atof(tmp);

			strncpy(tmp, str + 14, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->beta1 = atof(tmp);

			strncpy(tmp, str + 26, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->beta2 = atof(tmp);

			strncpy(tmp, str + 38, 12);
			tmp[12] = 0;
			replaceExpDesignator(tmp, 12);
			ionoutc->beta3 = atof(tmp);

			flags |= 0x1 << 1;
		}
		else if (strncmp(str + 60, "DELTA-UTC", 9) == 0)
		{
			strncpy(tmp, str + 3, 19);
			tmp[19] = 0;
			replaceExpDesignator(tmp, 19);
			ionoutc->A0 = atof(tmp);

			strncpy(tmp, str + 22, 19);
			tmp[19] = 0;
			replaceExpDesignator(tmp, 19);
			ionoutc->A1 = atof(tmp);

			strncpy(tmp, str + 41, 9);
			tmp[9] = 0;
			ionoutc->tot = atoi(tmp);

			strncpy(tmp, str + 50, 9);
			tmp[9] = 0;
			ionoutc->wnt = atoi(tmp);

			if (ionoutc->tot % 4096 == 0)
				flags |= 0x1 << 2;
		}
		else if (strncmp(str + 60, "LEAP SECONDS", 12) == 0)
		{
			strncpy(tmp, str, 6);
			tmp[6] = 0;
			ionoutc->dtls = atoi(tmp);

			flags |= 0x1 << 3;
		}
	}

	ionoutc->vflg = FALSE;
	if (flags == 0xF) // Read all Iono/UTC lines
		ionoutc->vflg = TRUE;

	// Read ephemeris blocks
	g0.week = -1;
	ieph = 0;

	while (1)
	{
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		// PRN
		strncpy(tmp, str, 2);
		tmp[2] = 0;
		sv = atoi(tmp) - 1;

		// EPOCH
		strncpy(tmp, str + 3, 2);
		tmp[2] = 0;
		t.y = atoi(tmp) + 2000;

		strncpy(tmp, str + 6, 2);
		tmp[2] = 0;
		t.m = atoi(tmp);

		strncpy(tmp, str + 9, 2);
		tmp[2] = 0;
		t.d = atoi(tmp);

		strncpy(tmp, str + 12, 2);
		tmp[2] = 0;
		t.hh = atoi(tmp);

		strncpy(tmp, str + 15, 2);
		tmp[2] = 0;
		t.mm = atoi(tmp);

		strncpy(tmp, str + 18, 4);
		tmp[2] = 0;
		t.sec = atof(tmp);

		date2gps(&t, &g);

		if (g0.week == -1)
			g0 = g;

		// Check current time of clock
		dt = subGpsTime(g, g0);

		if (dt > SECONDS_IN_HOUR)
		{
			g0 = g;
			ieph++; // a new set of ephemerides

			if (ieph >= EPHEM_ARRAY_SIZE)
				break;
		}

		// Date and time
		eph[ieph][sv].t = t;

		// SV CLK
		eph[ieph][sv].toc = g;//toc：GPS周GPS秒，即之后的信号发射时间

		strncpy(tmp, str + 22, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19); // tmp[15]='E';
		eph[ieph][sv].af0 = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].af1 = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].af2 = atof(tmp);

		// BROADCAST ORBIT - 1
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 3, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].iode = (int)atof(tmp);

		strncpy(tmp, str + 22, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].crs = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].deltan = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].m0 = atof(tmp);

		// BROADCAST ORBIT - 2
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 3, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].cuc = atof(tmp);

		strncpy(tmp, str + 22, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].ecc = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].cus = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].sqrta = atof(tmp);

		// BROADCAST ORBIT - 3
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 3, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].toe.sec = atof(tmp);//注意与TOC区别，TOE是和轨道相关的星历参数

		strncpy(tmp, str + 22, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].cic = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].omg0 = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].cis = atof(tmp);

		// BROADCAST ORBIT - 4
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 3, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].inc0 = atof(tmp);

		strncpy(tmp, str + 22, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].crc = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].aop = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].omgdot = atof(tmp);

		// BROADCAST ORBIT - 5
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 3, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].idot = atof(tmp);

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].toe.week = (int)atof(tmp);

		// BROADCAST ORBIT - 6
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		strncpy(tmp, str + 41, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].tgd = atof(tmp);

		strncpy(tmp, str + 60, 19);
		tmp[19] = 0;
		replaceExpDesignator(tmp, 19);
		eph[ieph][sv].iodc = (int)atof(tmp);

		// BROADCAST ORBIT - 7
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		// Set valid flag
		eph[ieph][sv].vflg = 1;

		// Update the working variables
		eph[ieph][sv].A = eph[ieph][sv].sqrta * eph[ieph][sv].sqrta;
		eph[ieph][sv].n = sqrt(GM_EARTH / (eph[ieph][sv].A*eph[ieph][sv].A*eph[ieph][sv].A)) + eph[ieph][sv].deltan;
		eph[ieph][sv].sq1e2 = sqrt(1.0 - eph[ieph][sv].ecc*eph[ieph][sv].ecc);
		eph[ieph][sv].omgkdot = eph[ieph][sv].omgdot - OMEGA_EARTH;
	}

	fclose(fp);

	if (g0.week >= 0)
		ieph += 1; // Number of sets of ephemerides

	return(ieph);
}

double ionosphericDelay(const ionoutc_t *ionoutc, gpstime_t g, double *llh, double *azel)
{
	double iono_delay = 0.0;
	double E, phi_u, lam_u, F;

	if (ionoutc->enable == FALSE)
		return (0.0); // No ionospheric delay

	E = azel[1] / PI;
	phi_u = llh[0] / PI;
	lam_u = llh[1] / PI;

	// Obliquity factor
	F = 1.0 + 16.0*pow((0.53 - E), 3.0);


	if (ionoutc->vflg == FALSE)
		iono_delay = F*5.0e-9*SPEED_OF_LIGHT;
	else
	{
		double t, psi, phi_i, lam_i, phi_m, phi_m2, phi_m3;
		double AMP, PER, X, X2, X4;

		// Earth's central angle between the user position and the earth projection of
		// ionospheric intersection point (semi-circles)
		psi = 0.0137 / (E + 0.11) - 0.022;

		// Geodetic latitude of the earth projection of the ionospheric intersection point
		// (semi-circles)
		phi_i = phi_u + psi*cos(azel[0]);
		if (phi_i > 0.416)
			phi_i = 0.416;
		else if (phi_i < -0.416)
			phi_i = -0.416;

		// Geodetic longitude of the earth projection of the ionospheric intersection point
		// (semi-circles)
		lam_i = lam_u + psi*sin(azel[0]) / cos(phi_i*PI);

		// Geomagnetic latitude of the earth projection of the ionospheric intersection
		// point (mean ionospheric height assumed 350 km) (semi-circles)
		phi_m = phi_i + 0.064*cos((lam_i - 1.617)*PI);
		phi_m2 = phi_m*phi_m;
		phi_m3 = phi_m2*phi_m;

		AMP = ionoutc->alpha0 + ionoutc->alpha1*phi_m
			+ ionoutc->alpha2*phi_m2 + ionoutc->alpha3*phi_m3;
		if (AMP < 0.0)
			AMP = 0.0;

		PER = ionoutc->beta0 + ionoutc->beta1*phi_m
			+ ionoutc->beta2*phi_m2 + ionoutc->beta3*phi_m3;
		if (PER < 72000.0)
			PER = 72000.0;

		// Local time (sec)
		t = SECONDS_IN_DAY / 2.0*lam_i + g.sec;
		while (t >= SECONDS_IN_DAY)
			t -= SECONDS_IN_DAY;
		while (t < 0)
			t += SECONDS_IN_DAY;

		// Phase (radians)
		X = 2.0*PI*(t - 50400.0) / PER;

		if (fabs(X) < 1.57)
		{
			X2 = X*X;
			X4 = X2*X2;
			iono_delay = F*(5.0e-9 + AMP*(1.0 - X2 / 2.0 + X4 / 24.0))*SPEED_OF_LIGHT;
		}
		else
			iono_delay = F*5.0e-9*SPEED_OF_LIGHT;
	}

	return (iono_delay);
}

/*! \brief Compute range between a satellite and the receiver
*  \param[out] rho The computed range
*  \param[in] eph Ephemeris data of the satellite
*  \param[in] g GPS time at time of receiving the signal
*  \param[in] xyz position of the receiver
*/
void computeRange(range_t *rho, ephem_t eph, ionoutc_t *ionoutc, gpstime_t g, double xyz[])
{
	double pos[3], vel[3], clk[2];
	double pos_t[3], vel_t[3], clk_t[2];
	double los[3];
	double tau;
	double range, rate;
	double xrot, yrot;

	double llh[3], neu[3];
	double tmat[3][3];
	double cw,sw;
	double speedoflight = 299792458.458;
	double tau_test;
	gpstime_t g_t=g;
	// SV position at time of the pseudorange observation.
	satpos(eph, g, pos, vel, clk);

	// Receiver to satellite vector and light-time.
	subVect(los, pos, xyz);//delta
	
	tau = normVect(los) / SPEED_OF_LIGHT;
	tau_test = tau;
	// Extrapolate the satellite position backwards to the transmission time.
	pos[0] -= vel[0] * tau;
	pos[1] -= vel[1] * tau;
	pos[2] -= vel[2] * tau;

	for (int ii = 0; ii < 5; ii++) //使用迭代的方法精度得到提高
	{
		g_t.sec = g.sec - tau_test;//得到发射时间；
		satpos(eph, g_t, pos_t, vel_t, clk_t);
		cw = cos(OMEGA_EARTH * tau_test);
		sw = sin(OMEGA_EARTH * tau_test);
		pos_t[0] = cw * pos_t[0] + sw * pos_t[1];
		pos_t[1] = -sw * pos_t[0] + cw * pos_t[1];
		subVect(los, pos_t, xyz);
		tau_test = normVect(los) / speedoflight;
	}

	// Earth rotation correction. The change in velocity can be neglected.
	xrot = pos[0] + pos[1] * OMEGA_EARTH*tau;//修正地球转动带来的影响
	yrot = pos[1] - pos[0] * OMEGA_EARTH*tau;
	pos[0] = xrot;
	pos[1] = yrot;

	// New observer to satellite vector and satellite range.
	subVect(los, pos, xyz);
	range = normVect(los);
	rho->d = range;//在接收时刻卫星和接收机的距离
	tau = range / SPEED_OF_LIGHT;

	// Pseudorange.
	rho->range = range - SPEED_OF_LIGHT*clk[0];//距离减去钟差

	// Relative velocity of SV and receiver.
	rate = dotProd(vel, los) / range;//卫星和接收机之间的相对速度标量

	// Pseudorange rate.
	rho->rate = rate; // - SPEED_OF_LIGHT*clk[1];

					  // Time of application.
	rho->g = g;

	// Azimuth and elevation angles.
	xyz2llh(xyz, llh);
	ltcmat(llh, tmat);
	ecef2neu(los, tmat, neu);
	neu2azel(rho->azel, neu);

	// Add ionospheric delay
	rho->iono_delay = ionosphericDelay(ionoutc, g, llh, rho->azel) / 0.6;
	rho->range += rho->iono_delay;//电离层延迟对码相位是延迟作用
	return;
}

/*! \brief Compute the code phase for a given channel (satellite)
*  \param chan Channel on which we operate (is updated)
*  \param[in] rho1 Current range, after \a dt has expired
*  \param[in dt delta-t (time difference) in seconds
*/
void computeCodePhase(channel_t *chan, range_t rho1, double dt)
{
	double ms;
	int ims;
	double rhorate;

	// Pseudorange rate.
  	rhorate = (rho1.range - chan->rho0.range) / dt;//距离差除时间差得到平均速度//得到径向速度

	// Carrier and code frequency.
	chan->f_carr = -rhorate / LAMBDA_L1;//f_carr就是多普勒频移，由于生成的是零中频信号，载波偏移就是多普勒频移fc=-v/c*f 未考虑钟飘修正项？
	chan->f_code = CODE_FREQ + chan->f_carr*CARR_TO_CODE;

	// Initial code phase and data bit counters.
	double a = chan->rho0.g.sec-chan->g0.sec;
 	ms = ((subGpsTime(chan->rho0.g, chan->g0) + 6.0) - chan->rho0.range / SPEED_OF_LIGHT)*1000.0;//6s是一个子帧的时间,避免减去传播时间后出现负数，算到上一子帧

	ims = (int)ms;
	chan->code_phase = (ms - (double)ims)*CA_SEQ_LEN; // in chip

	chan->iword = ims / 600; // 1 word = 30 bits = 600 ms
	ims -= chan->iword  * 600;

	chan->ibit = ims / 20; // 1 bit = 20 code = 20 ms//时间为第一帧末尾？
	ims -= chan->ibit * 20;

	chan->icode = ims; // 1 code = 1 ms

	chan->codeCA = chan->ca[(int)chan->code_phase] * 2 - 1;
	chan->dataBit = (int)((chan->dwrd[chan->iword] >> (29 - chan->ibit)) & 0x1UL) * 2 - 1;//发的是上一帧的第五子帧末尾

	// Save current pseudorange
	chan->rho0 = rho1;

	return;
}

/*! \brief Read the list of user motions from the input file
*  \param[out] xyz Output array of ECEF vectors for user motion
*  \param[[in] filename File name of the text input file
*  \returns Number of user data motion records read, -1 on error
*/
int readUserMotion(double (*xyz)[3], const char *filename,int tsim)
{
	FILE *fp;
	int numd;
	char str[MAX_CHAR];
	double t, x, y, z;
	if (NULL == (fp = fopen(filename, "rt")))
		return(-1);

	for (numd = 0; numd <tsim ; numd++)
	{
		if (fgets(str, MAX_CHAR, fp) == NULL)
			break;

		if (EOF == sscanf(str, "%lf,%lf,%lf,%lf", &t, &x, &y, &z)) // Read CSV line
			break;

		xyz[numd][0] = x;
		xyz[numd][1] = y;
		xyz[numd][2] = z;
	}

	fclose(fp);

	return (numd);
}

int readNmeaGGA(double (*xyz)[3], const char *filename)
{
	FILE *fp;
	int numd = 0;
	char str[MAX_CHAR];
	char *token;
	double llh[3], pos[3];
	char tmp[8];

	if (NULL == (fp = fopen(filename, "rt")))
		return(-1);

	while (1)
	{
		if (fgets(str, MAX_CHAR, fp) == NULL)
			break;

		token = strtok(str, ",");

		if (strncmp(token + 3, "GGA", 3) == 0)
		{
			token = strtok(NULL, ","); // Date and time

			token = strtok(NULL, ","); // Latitude
			strncpy(tmp, token, 2);
			tmp[2] = 0;

			llh[0] = atof(tmp) + atof(token + 2) / 60.0;

			token = strtok(NULL, ","); // North or south
			if (token[0] == 'S')
				llh[0] *= -1.0;

			llh[0] /= R2D; // in radian

			token = strtok(NULL, ","); // Longitude
			strncpy(tmp, token, 3);
			tmp[3] = 0;

			llh[1] = atof(tmp) + atof(token + 3) / 60.0;

			token = strtok(NULL, ","); // East or west
			if (token[0] == 'W')
				llh[1] *= -1.0;

			llh[1] /= R2D; // in radian

			token = strtok(NULL, ","); // GPS fix
			token = strtok(NULL, ","); // Number of satellites
			token = strtok(NULL, ","); // HDOP

			token = strtok(NULL, ","); // Altitude above meas sea level

			llh[2] = atof(token);

			token = strtok(NULL, ","); // in meter

			token = strtok(NULL, ","); // Geoid height above WGS84 ellipsoid

			llh[2] += atof(token);

			// Convert geodetic position into ECEF coordinates
			llh2xyz(llh, pos);

			xyz[numd][0] = pos[0];
			xyz[numd][1] = pos[1];
			xyz[numd][2] = pos[2];

			// Update the number of track points
			numd++;

			if (numd >= USER_MOTION_SIZE)
				break;
		}
	}

	fclose(fp);

	return (numd);
}

int generateNavMsg(gpstime_t g, channel_t *chan, int init,char * charbit)
{
	int iwrd, isbf;
	gpstime_t g0;
	unsigned long wn, tow;
	unsigned sbfwrd;
	unsigned long prevwrd;
	int nib;
	char* str = NULL;
	str = (char*)malloc(10);
	//写电文部分
	/*FILE* feph;
	sprintf(str, "%dEPH.txt", chan->prn);
	feph=fopen(str, "wt+");*/

	g0.week = g.week;
	g0.sec = (double)(((unsigned long)(g.sec + 0.5)) / 30UL) * 30.0; // Align with the full frame length = 30 sec
	chan->g0 = g0; // Data bit reference time

	wn = (unsigned long)(g0.week % 1024);
	tow = ((unsigned long)g0.sec) / 6UL;//一个子帧持续6s,一个子帧有一个tow位，所以需要除6s

	if (init == 1) // Initialize subframe 5 
	{
		prevwrd = 0UL;

		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)//存储第五子帧的10个字
		{
			sbfwrd = chan->sbf[4][iwrd];//为什么先存入的是第五子帧

			// Add TOW-count message into HOW
			if (iwrd == 1)
				sbfwrd |= ((tow & 0x1FFFFUL) << 13);

			// Compute checksum
			sbfwrd |= ( prevwrd << 30) & 0xC0000000UL; // 2 LSBs of the previous transmitted word 保留高两位，其余位给零，prevwrd左移30位写入sbfwrd中
			nib = ((iwrd == 1) || (iwrd == 9)) ? 1 : 0; // Non-information bearing bits for word 2 and 10 这是什么意思
			chan->dwrd[iwrd] = computeChecksum(sbfwrd, nib);

			prevwrd = chan->dwrd[iwrd];
		}
	}
	else // Save subframe 5
	{
		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)
		{
			chan->dwrd[iwrd] = chan->dwrd[N_DWRD_SBF*N_SBF + iwrd];

			prevwrd = chan->dwrd[iwrd];
		}
		/*
		// Sanity check
		if (((chan->dwrd[1])&(0x1FFFFUL<<13)) != ((tow&0x1FFFFUL)<<13))
		{
		printf("\nWARNING: Invalid TOW in subframe 5.\n");
		return(0);
		}
		*/
	}
	//前十个字是上一帧的第五子帧，后五十个字是这一帧的五个子帧
	for (isbf = 0; isbf < N_SBF; isbf++)//每帧有5个子帧 
	{
		tow++;//每一个子帧都包含一个TOW

		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)//每个子帧包含十个字
		{
			sbfwrd = chan->sbf[isbf][iwrd];

			// Add transmission week number to Subframe 1
			if ((isbf == 0) && (iwrd == 2))
				sbfwrd |= (wn & 0x3FFUL) << 20;

			// Add TOW-count message into HOW
			if (iwrd == 1)
				sbfwrd |= ((tow & 0x1FFFFUL) << 13);

			// Compute checksum
			sbfwrd |= (prevwrd << 30) & 0xC0000000UL; // 2 LSBs of the previous transmitted word
			nib = ((iwrd == 1) || (iwrd == 9)) ? 1 : 0; // Non-information bearing bits for word 2 and 10
			chan->dwrd[(isbf + 1)*N_DWRD_SBF + iwrd] = computeChecksum(sbfwrd, nib);

			prevwrd = chan->dwrd[(isbf + 1)*N_DWRD_SBF + iwrd];//子帧数乘十加当前word数
		}
	}
	//写电文部分
	/*for (int jj = 0; jj < 50; jj++)
	{
		if (jj % 10 == 0&&jj>0)
		{
			fprintf(feph, "\n");
		}
		fprintf(feph, "%#010x ", chan->dwrd[jj + 10]);
		
	}
	fclose(feph);*/

	int wrd = 0, bit = 0;
	for (int cnt = 0; cnt < 1800; cnt++,bit++)//将二进制电文转换成为char型的数组，其实应该是unsigned char
	{
		if (bit == 30)
		{
			bit = 0; wrd++;
		}
		charbit[cnt + (chan->prn - 1) * 1800] = (char)((chan->dwrd[wrd] >> (29 - bit)) & 0x1UL) * 2 - 1;
	}
	//for (int jj = 0; jj < 60; jj++)
	//{
	//	for (int kk = 0; kk < 30; kk++)
	//	{
	//		printf("%d ",charbit[jj * 30 + kk + (chan->prn - 1) * 1800]);
	//	}
	//	printf("\n");
	//}
	return(1);
}


int checkSatVisibility(ephem_t eph, gpstime_t g, double *xyz, double elvMask, double *azel)
{
	double llh[3], neu[3];
	double pos[3], vel[3], clk[3], los[3];
	double tmat[3][3];

	if (eph.vflg != 1)
		return (-1); // Invalid

	xyz2llh(xyz, llh);
	ltcmat(llh, tmat);

	satpos(eph, g, pos, vel, clk);
	subVect(los, pos, xyz);
	ecef2neu(los, tmat, neu);
	neu2azel(azel, neu);

	if (azel[1] * R2D > elvMask)
		return (1); // Visible
					// else
	return (0); // Invisible
}
/// <summary>
/// 给可见卫星分配通道
/// </summary>
/// <param name="chan"></param>
/// <param name="eph"></param> 当前两小时星历
/// <param name="ionoutc"></param>
/// <param name="grx"></param>
/// <param name="xyz"></param>
/// <param name="elvMaskc"></param>
/// <param name="navbit"></param>
/// <returns></returns>
int allocateChannel(channel_t *chan, ephem_t *eph, ionoutc_t ionoutc, gpstime_t grx, double *xyz, double elvMaskc,char *navbit, int allocatedSat[])//
{
	int nsat = 0;
	int i, sv;
	double azel[2];

	range_t rho;
	double ref[3] = { 0.0 };
	double r_ref, r_xyz;
	double phase_ini;
	FILE* fwph = NULL;
	int n = 0;


	for (sv = 0; sv < MAX_SAT; sv++)
	{
		if (checkSatVisibility(eph[sv], grx, xyz, 0.0, azel) == 1)
		{
			nsat++; // Number of visible satellites 

			if (allocatedSat[sv] == -1) // Visible but not allocated  =-1
			{
				// Allocated new satellite.
				for (i = 0; i < MAX_CHAN; i++)//
				{
					if (chan[i].prn == 0)
					{
						// Initialize channel
						chan[i].prn = sv + 1;
						chan[i].azel[0] = azel[0];
						chan[i].azel[1] = azel[1];

						// C/A code generation
						codegen(chan[i].ca, chan[i].prn);

						// Generate subframe
						eph2sbf(eph[sv], ionoutc, chan[i].sbf);

						// Generate navigation message
						generateNavMsg(grx, &chan[i], 1,navbit);

						// Initialize pseudorange 
						computeRange(&rho, eph[sv], &ionoutc, grx, xyz);//包含的伪距信息
						chan[i].rho0 = rho;

						// Initialize carrier phase
						r_xyz = rho.range;//获得伪距信息

						computeRange(&rho, eph[sv], &ionoutc, grx, ref);//
 						r_ref = rho.range;

						phase_ini = (2.0*r_ref - r_xyz) / LAMBDA_L1;//在接收时刻接收机和卫星的距离，地心原点与卫星距离
						phase_ini -= floor(phase_ini);
						//chan[i].carr_phase = (unsigned int)(512 * 65536.0 * phase_ini);//*65536将载波相位变为整型，*512相当于逆归一化过程
						chan[i].carr_phase = phase_ini;
						// Done.
						break;
					}
				}
				// Set satellite allocation channel
				if (i < MAX_CHAN)
					allocatedSat[sv] = i;
			}
		}
		else if (allocatedSat[sv] >= 0) // Not visible but allocated
		{
			// Clear channel
			chan[allocatedSat[sv]].prn = 0;

			// Clear satellite allocation flag
			allocatedSat[sv] = -1;
		}
	}

	return(nsat);
}

void usage(void)
{
	printf("Usage: gps-sdr-sim [options]\n"
		"Options:\n"
		"  -e <gps_nav>     RINEX navigation file for GPS ephemerides (required)\n"
		"  -u <user_motion> User motion file (dynamic mode)\n"
		"  -g <nmea_gga>    NMEA GGA stream (dynamic mode)\n"
		"  -l <location>    Lat,Lon,Hgt (static mode) e.g. 30.286502,120.032669,100\n"
		"  -t <date,time>   Scenario start time YYYY/MM/DD,hh:mm:ss\n"
		"  -T <date,time>   Overwrite TOC and TOE to scenario start time\n"
		"  -d <duration>    Duration [sec] (max: %.0f)\n"
		"  -o <output>      I/Q sampling data file (default: gpssim.bin)\n"
		"  -s <frequency>   Sampling frequency [Hz] (default: 2600000)\n"
		"  -b <iq_bits>     I/Q data format [1/8/16] (default: 16)\n"
		"  -i               Disable ionospheric delay for spacecraft scenario\n"
		"  -v               Show details about simulated channels\n",
		((double)USER_MOTION_SIZE) / 10.0);

	return;
}

void zeros(int *am, int start, int endv) {
	int i, a = endv - start;
	if (a > 0) {
		for (i = 0; i < a; i++) {
			am[i] = 0;
		}
	}
}

int isEmpty(char *a) {
	if (a == NULL || strcmp(a, "") == 0)
	{
		return 1;
	}

	return 0;
}

int checkParamreq(char *navfn) {
	return !isEmpty(navfn);
}
int check_can_run(char *navfn, char *user_motion, char *nmea_gga, char *localtion, char *datetime, char *datetime_toc_toe, char *duration, char *outputfile, char *frequecy, char *iqbit) {
	int a[9] = { 0 };
	int i, dem = 0;
	if (checkParamreq(navfn))
	{
		if (!isEmpty(user_motion))a[0] = 1;
		if (!isEmpty(nmea_gga))a[1] = 1;
		if (!isEmpty(localtion))a[2] = 1;
		if (!isEmpty(datetime))a[3] = 1;
		if (!isEmpty(datetime_toc_toe))a[4] = 1;
		if (!isEmpty(duration))a[5] = 1;
		if (!isEmpty(outputfile))a[6] = 1;
		if (!isEmpty(frequecy))a[7] = 1;
		if (!isEmpty(iqbit))a[8] = 1;
		for (i = 0; i < 9; i++)
		{
			if (a[i] == 1)
				dem++;
		}
		if (dem > 0)return 1;
	}
	return 0;
}

inline//内联函数，类似于宏展开
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s error code %d\n", cudaGetErrorString(result), result);
		getchar();
		//assert(result == cudaSuccess);
	}
#endif
	return result;
}
	 
float gps_sim(Table *look_up,transfer_parameter *tp, complexf* buff,size_t max_len,int freq_samp,int fc,int tsim, float* dev_i_buff)
{
	//clock_t tstart, tend, tstart1, tEnd1, sumTime;
	//char navfn[], char user_motion[], char nmea_gga[], char localtion[], char datetime[], char datetime_toc_toe[], char duration1[], char outputfile[], char frequecy[], char iqbit[]
	//char *navfn=NULL;  //D:\\PL\\GR_GPS_Cuda_mix\\GPSL1_sim_quantify\\brdc2017_0660.17n  brdc3540.14n
	//char *user_motion= NULL;        //circle1.csv
	static int count_call=0;
	char nmea_gga[256] = { 0 };
	char localtion[256] = { 0 };
	char datetime[256] = { 0 };
	char datetime_toc_toe[256] = { 0 };
	char duration1[256] = { 0 };
	char outputfile[256] = { 0 };
	char frequecy[256] = { 0 };
	char iqbit[256] = { 0 };
	char buffer[256] = { 0 };

	int bcount = 0, rcount = 0;
	int loop_count = 0;
	unsigned char ibit = 0;

	FILE *fp;
	short *a, *a1, *b, *c1, *c, *d;
	int sv;
	 static int count;//代表卫星数量，必须是一个静态变量
	static int  ieph;//代表当前选中的星历数，必须是一个静态变量
	//ephem_t eph[EPHEM_ARRAY_SIZE][MAX_SAT];//一帧30s,一共13帧，每两小时星历更改一次，13帧覆盖24h的信号
	//gpstime_t g0;

	double llh[3];

	int i;
	//channel_t chan[MAX_CHAN];
	double elvmask = 0.0; // in degree

						  //short ip,qp;
	int ip, qp;
	short *iq_buff = NULL;
	signed char *iq8_buff = NULL;

	//gpstime_t grx;
	double delt;
	int isamp;

	int iumd;
	int numd;
	char umfile[MAX_CHAR];
	const int const_tsim = tsim;
	//double (*xyz)[3] = NULL;
	//xyz =(double(*)[3]) malloc(sizeof(double) * tsim * 3);

	int staticLocationMode = FALSE;
	int nmeaGGA = FALSE;

	char navfile[MAX_CHAR];
	char outfile[MAX_CHAR];

	double samp_freq;
	int iq_buff_size;
	int data_format;


	float gain[MAX_CHAN];
	double path_loss;
	double ant_gain;
	double ant_pat[37];
	int ibs; // boresight angle index

	//datetime_t t0, tmin, tmax;
	//gpstime_t gmin, gmax;
	double dt;
	int igrx;

	double duration;
	int iduration;
	int verb;


	//char* navbit = NULL; //将二进制导航电文存为char型数组形式

	int timeoverwrite = FALSE; // Overwirte the TOC and TOE in the RINEX file

	//ionoutc_t ionoutc;

	////////////////////////////////////////////////////////////
	// Read options
	////////////////////////////////////////////////////////////

	// Default options
	//if (rinex_file != 0)
	//{
	//	navfn = (char*)malloc(strlen(rinex_file)+1);//如果不加1，free的时候会报错
	//	strcpy(navfn, rinex_file);
	//}
	//if (traj_file !=0)
	//{
	//	user_motion = (char*)malloc(strlen(traj_file)+1);
	//	strcpy(user_motion, traj_file);
	//}
	/*navfile[0] = 0;
	umfile[0] = 0;*/
	strcpy(outfile, "gpssim.bin");
	samp_freq = freq_samp;
	data_format = SC16;
	//g0.week = -1; // Invalid start time
	iduration = tsim;
	verb = FALSE;
	tp->ionoutc.enable = TRUE;
	/*strcpy(navfile, navfn);
	if (isEmpty(user_motion) == 0) {
		strcpy(umfile, user_motion);
		nmeaGGA = FALSE;
	}
	if (isEmpty(nmea_gga) == 0) {
		strcpy(umfile, nmea_gga);
		nmeaGGA = TRUE;
	}*/

	if (isEmpty(localtion) == 0) {
		// Static geodetic coordinates input mode
		// Added by scateu@gmail.com
		staticLocationMode = TRUE;
		sscanf(localtion, "%lf,%lf,%lf", &llh[0], &llh[1], &llh[2]);
		llh[0] = llh[0] / R2D; // convert to RAD
		llh[1] = llh[1] / R2D; // convert to RAD
	}

	if (isEmpty(outputfile) == 0) {
		strcpy(outfile, outputfile);
	}
	
	if (isEmpty(datetime_toc_toe) == 0) {
		timeoverwrite = TRUE;
		if (strncmp(datetime_toc_toe, "now", 3) == 0)
		{
			time_t timer;
			struct tm *gmt;

			time(&timer);
			gmt = gmtime(&timer);

			tp->t0.y = gmt->tm_year + 1900;
			tp->t0.m = gmt->tm_mon + 1;
			tp->t0.d = gmt->tm_mday;
			tp->t0.hh = gmt->tm_hour;
			tp->t0.mm = gmt->tm_min;
			tp->t0.sec = (double)gmt->tm_sec;

			date2gps(&(tp->t0), &(tp->g0));

		}
	}
	if (isEmpty(datetime) == 0) {
		sscanf(datetime, "%d/%d/%d,%d:%d:%lf", &tp->t0.y, &tp->t0.m, &tp->t0.d, &tp->t0.hh, &tp->t0.mm, &tp->t0.sec);
		if (tp->t0.y <= 1980 || tp->t0.m < 1 || tp->t0.m>12 || tp->t0.d < 1 || tp->t0.d>31 ||
			tp->t0.hh < 0 || tp->t0.hh>23 || tp->t0.mm < 0 || tp->t0.mm>59 || tp->t0.sec < 0.0 || tp->t0.sec >= 60.0)
		{
			printf("ERROR: Invalid date and time.\n");
			exit(1);
		}
		tp->t0.sec = floor(tp->t0.sec);
		date2gps(&(tp->t0), &(tp->g0));
	}
	if (isEmpty(duration1) == 0) {
		duration = atof(duration1);
		if (duration<0.0 || duration>((double)tsim) / 10.0)
		{
			printf("ERROR: Invalid duration.\n");
			exit(1);
		}
		iduration = (int)(duration*10.0 + 0.5);
	}


	// Buffer size	
	samp_freq = floor(samp_freq / Rev_fre);
	iq_buff_size = (int)samp_freq; // samples per 0.1sec
	samp_freq *= Rev_fre; 

	delt = 1.0 / samp_freq;

	////////////////////////////////////////////////////////////
	// Receiver position
	////////////////////////////////////////////////////////////

	numd = iduration;
	/*
	printf("xyz = %11.1f, %11.1f, %11.1f\n", xyz[0][0], xyz[0][1], xyz[0][2]);
	printf("llh = %11.6f, %11.6f, %11.1f\n", llh[0]*R2D, llh[1]*R2D, llh[2]);
	*/
	////////////////////////////////////////////////////////////
	// Read ephemeris
	////////////////////////////////////////////////////////////

	//neph = readRinexNavAll(eph, &ionoutc, navfile);  //读取rinex，返回星数？
	if (count_call == 0)
	{
		if (tp->neph == 0)
		{
			printf("ERROR: No ephemeris available.\n");
			exit(1);
		}

		if ((verb == TRUE) && (tp->ionoutc.vflg == TRUE))
		{
			printf("  %12.3e %12.3e %12.3e %12.3e\n",
				tp->ionoutc.alpha0, tp->ionoutc.alpha1, tp->ionoutc.alpha2, tp->ionoutc.alpha3);
			printf("  %12.3e %12.3e %12.3e %12.3e\n",
				tp->ionoutc.beta0, tp->ionoutc.beta1, tp->ionoutc.beta2, tp->ionoutc.beta3);
			printf("   %19.11e %19.11e  %9d %9d\n",
				tp->ionoutc.A0, tp->ionoutc.A1, tp->ionoutc.tot, tp->ionoutc.wnt);
			printf("%6d\n", tp->ionoutc.dtls);
		}

		for (sv = 0; sv < MAX_SAT; sv++)
		{
			if (tp->eph[0][sv].vflg == 1)
			{
				tp->gmin = tp->eph[0][sv].toc;
				tp->tmin = tp->eph[0][sv].t;
				break;
			}
		}

		for (sv = 0; sv < MAX_SAT; sv++)
		{
			if (tp->eph[tp->neph - 1][sv].vflg == 1)
			{
				tp->gmax = tp->eph[tp->neph - 1][sv].toc;
				tp->tmax = tp->eph[tp->neph - 1][sv].t;         //最大运行时间
				break;
			}
		}

		if (tp->g0.week >= 0) // Scenario start time has been set.
		{
			if (timeoverwrite == TRUE)
			{
				gpstime_t gtmp;
				datetime_t ttmp;
				double dsec;

				gtmp.week = tp->g0.week;
				gtmp.sec = (double)(((int)(tp->g0.sec)) / 7200) * 7200.0;

				dsec = subGpsTime(gtmp, tp->gmin);

				// Overwrite the UTC reference week number
				tp->ionoutc.wnt = gtmp.week;
				tp->ionoutc.tot = (int)gtmp.sec;

				// Iono/UTC parameters may no longer valid
				//tp->ionoutc.vflg = FALSE;

				// Overwrite the TOC and TOE to the scenario start time
				for (sv = 0; sv < MAX_SAT; sv++)
				{
					for (i = 0; i < tp->neph; i++)
					{
						if (tp->eph[i][sv].vflg == 1)
						{
							gtmp = incGpsTime(tp->eph[i][sv].toc, dsec);
							gps2date(&gtmp, &ttmp);
							tp->eph[i][sv].toc = gtmp;
							tp->eph[i][sv].t = ttmp;

							gtmp = incGpsTime(tp->eph[i][sv].toe, dsec);
							tp->eph[i][sv].toe = gtmp;
						}
					}
				}
			}
			else
			{
				if (subGpsTime(tp->g0, tp->gmin) < 0.0 || subGpsTime(tp->gmax, tp->g0) < 0.0)
				{
					printf("ERROR: Invalid start time.\n");
					printf("tmin = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
						tp->tmin.y, tp->tmin.m, tp->tmin.d, tp->tmin.hh, tp->tmin.mm, tp->tmin.sec,
						tp->gmin.week, tp->gmin.sec);
					printf("tmax = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
						tp->tmax.y, tp->tmax.m, tp->tmax.d, tp->tmax.hh, tp->tmax.mm, tp->tmax.sec,
						tp->gmax.week, tp->gmax.sec);
					exit(1);
				}
			}
		}
		else
		{
			tp->g0 = tp->gmin;     //设置当前接收机时间为gmin
			tp->t0 = tp->tmin;
		}
		printf("Start time = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
			tp->t0.y, tp->t0.m, tp->t0.d, tp->t0.hh, tp->t0.mm, tp->t0.sec, tp->g0.week, tp->g0.sec);
		printf("Duration = %.1f [sec]\n", ((double)numd) / 10.0);

		// Select the current set of ephemerides
		ieph = -1;

		for (i = 0; i < tp->neph; i++)
		{
			for (sv = 0; sv < MAX_SAT; sv++)
			{
				if (tp->eph[i][sv].vflg == 1)
				{
					dt = subGpsTime(tp->g0, tp->eph[i][sv].toc);
					if (dt >= -SECONDS_IN_HOUR && dt < SECONDS_IN_HOUR)//一个卫星轨道星历的有效时间是toe前后一小时
					{
						ieph = i;
						break;
					}
				}
			}

			if (ieph >= 0) // ieph has been set
				break;
		}

		if (ieph == -1)
		{
			printf("ERROR: No current set of ephemerides has been found.\n");
			exit(1);
		}

		////////////////////////////////////////////////////////////
		// Baseband signal buffer and output file
		////////////////////////////////////////////////////////////

		// Allocate I/Q buffer
		iq_buff = new short[2 * iq_buff_size];

		if (data_format == SC08)
		{
			iq8_buff = (signed char*)calloc(2 * iq_buff_size, 2);    //calloc(2 * iq_buff_size, 1)？
			//iq8_buff = new signed char[2 * iq_buff_size];
			if (iq8_buff == NULL)
			{
				printf("ERROR: Faild to allocate 8-bit I/Q buffer.\n");
				exit(1);
			}
		}
		else if (data_format == SC01)
		{
			iq8_buff = (signed char*)calloc(iq_buff_size / 4, 1);
			//iq8_buff = new  signed char[iq_buff_size / 4]; // byte = {I0, Q0, I1, Q1, I2, Q2, I3, Q3}
			if (iq8_buff == NULL)
			{
				printf("ERROR: Faild to allocate compressed 1-bit I/Q buffer.\n");
				exit(1);
			}
		}

		// Open output file

		////////////////////////////////////////////////////////////
		// Initialize channels
		////////////////////////////////////////////////////////////


		// Clear all channels
		for (i = 0; i < MAX_CHAN; i++)
			tp->chan[i].prn = 0;//通道失能是0，prn是从1开始的

		// Clear satellite allocation flag
		for (sv = 0; sv < MAX_SAT; sv++)
			tp->allocatedSat[sv] = -1;//卫星是否分配通道标志 default:-1

		// Initial reception time
		tp->grx = incGpsTime(tp->g0, 0.0);//初始化接收时间x,设置为和g0一样的时间

		// Allocate visible satellites
		allocateChannel(tp->chan, tp->eph[ieph], tp->ionoutc, tp->grx, tp->xyz[0], elvmask, tp->navbit, tp->allocatedSat);

		count = 0;
		for (i = 0; i < MAX_CHAN; i++)
		{
			if (tp->chan[i].prn > 0)
			{
				printf("%02d %6.1f %5.1f %11.1f %5.1f\n", tp->chan[i].prn,
					tp->chan[i].azel[0] * R2D, tp->chan[i].azel[1] * R2D, tp->chan[i].rho0.d, tp->chan[i].rho0.iono_delay);
				count++;
			}
		}
		tp->grx = incGpsTime(tp->grx, 0.1);//接收机时间增加0.1s
		////////////////////////////////////////////////////////////
		// Receiver antenna gain pattern
		////////////////////////////////////////////////////////////

		//for (i = 0; i < 37; i++)
		//	ant_pat[i] = mpow(10.0, -ant_pat_db[i] / 20.0);//对以db为单位的增益进行转换

		////////////////////////////////////////////////////////////
		// Generate baseband signals
		////////////////////////////////////////////////////////////
	}
	float* noise;
	double rr = 1000;
	for (i = 0; i < MAX_CHAN; i++)
	{
		gain[i] = sqrt(2 * pow(10.0, (5.0 + 0.1 * i)) / samp_freq);
	}

	noise = (float*)malloc(2 * iq_buff_size * sizeof(float));

	for (i = 0; i < 2 * iq_buff_size; i++)
	{
		noise[i] = (float)mgrn1(0, 1, &rr);//均值为0，方差为1
	}
	//for (i = 5e5-200; i < 5e5; i++)
	//{
	//	printf("%f %f\n", noise[2 * i], noise[2 * i + 1]);
	//}
	//int bpoint = 5;
	// Update receiver time

	//for (isamp = 0; isamp < iq_buff_size; isamp++)
	//{
	//	count = 0;
	//	for (int i = 0; i < MAX_CHAN; i++)
	//	{
	//		if (chan[i].prn > 0)
	//			count++;
 //		}

	//}

	//a  = (short*)malloc(count * iq_buff_size * sizeof(short));//13*1e6

	//b = (short*)malloc(count * iq_buff_size * sizeof(short));
	//c = (short*)malloc(count * iq_buff_size * sizeof(short));
	//c1 = (short*)malloc(count * iq_buff_size * sizeof(short));


	//point = (TPoint *)malloc(count * sizeof(TPoint));

	

	//free(CAcode);


	look_up->cosTable= &cosTable512[0];
	look_up->sinTable= &sinTable512[0];
	look_up->navdata = tp->navbit;
	//GPSL1table.qua_buff = (int*)malloc(sizeof(int) * iq_buff_size / 5);
	GPUMemoryInit(look_up,count);
	//申请核函数中使用的变量
	int  * dev_sum, * sum;
	float* parameters, * dev_parameters,*dev_noise;
	double* para_doub, * dev_para_doub;
	checkCuda(cudaHostAlloc((void**)&dev_noise, iq_buff_size * sizeof(float) * 2, cudaHostAllocDefault));
	checkCuda(cudaMalloc((void**)&dev_sum, 2 * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_parameters, p_n * count * sizeof(float)));//存放数据的含义可以见kernel文件
	checkCuda(cudaMemcpy(dev_noise, noise, sizeof(float) * iq_buff_size * 2, cudaMemcpyHostToDevice));
	checkCuda(cudaMalloc((void**)&dev_para_doub, pd_n * count * sizeof(double)));
	//主机变量的内存分配采样页锁定内存
	checkCuda(cudaHostAlloc((void**)&parameters, sizeof(float)* count* p_n * 10,cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&para_doub, sizeof(double)* count* p_n * 10, cudaHostAllocDefault));
	checkCuda(cudaHostAlloc((void**)&sum, sizeof(int) * 2, cudaHostAllocDefault));

	/*parameters = (float*)malloc(sizeof(float) * count * p_n);
	para_doub = (double*)malloc(sizeof(double) * count * pd_n);
	sum = (int*)malloc(sizeof(int) * 2);*/
	clock_t tstart = clock();


	//启动计时器
	cudaEvent_t starttext, stop;
	float elapsedTime;
	checkCuda(cudaEventCreate(&starttext));
	checkCuda(cudaEventCreate(&stop));
	checkCuda(cudaEventRecord(starttext, 0));


	float temp = 0;
	int start;
	if (count_call == 0)
	{
		start = 1;//第一个0.1s的数据不产生start = 1;
	}
	else
	{
		start = 0;
	}
	for (iumd = start; iumd < numd ; iumd++)           //开始每历元，每个通道的循环
	{
		printf("\nlap = %d", iumd+ count_call* numd);
		loop_count += 1;
		for (i = 0; i < MAX_CHAN; i++)//每次都是重新计算码相位载波相位和步进
		{
			if (tp->chan[i].prn > 0)
			{
				// Refresh code phase and data bit counters
				range_t rho;
				sv = tp->chan[i].prn - 1;

				// Current pseudorange
				computeRange(&rho, tp->eph[ieph][sv], &tp->ionoutc, tp->grx, tp->xyz[iumd + count_call * numd]);//在grx时刻下接收机和卫星距离
				tp->chan[i].azel[0] = rho.azel[0];
				tp->chan[i].azel[1] = rho.azel[1];

				// Update code phase and data bit counters
				computeCodePhase(&(tp->chan[i]), rho, 0.1);
				tp->chan[i].carr_phasestep = (tp->chan[i].f_carr+fc) * delt;//增加了中频
				tp->chan[i].code_phasestep = tp->chan[i].f_code / samp_freq;
				// Path loss
				path_loss = 20200000.0 / rho.d;//20200000 为轨道高度 rho.d
				// Receiver antenna gain
				ibs = (int)((90.0 - rho.azel[1] * R2D) / 5.0); // covert elevation to boresight
				tp->chan[i].amp = gain[i];
			}
		}

		//produce_samples_withCuda(look_up, &(tp->chan[0]), samp_freq,parameters, dev_parameters, sum, dev_sum, dev_i_buff, count,dev_noise, para_doub,dev_para_doub);
		Storedata(&(tp->chan[0]), count_call,iumd, sum, count, samp_freq, parameters, para_doub);

		for (int satid = 0; satid < MAX_CHAN; satid++)
		{
			tp->chan[satid].carr_phase += tp->chan[satid].carr_phasestep * iq_buff_size;
			tp->chan[satid].carr_phase -= int(tp->chan[satid].carr_phase);
		}
		/*for (int jj = 0; jj < iq_buff_size; jj++)
		{
			buff->real(look_up->i_buff[jj]);
			buff->imag(look_up->i_buff[jj + iq_buff_size]);
			buff++;
		}*/
		//
		// Update navigation message and channel allocation every 30 seconds
		//

		igrx = (int)(tp->grx.sec*10.0 + 0.5);//grx.sec在乘10后很可能是.099999，需要+0.5再取整，否则可能减少了一个历元的数据

		if (igrx % 300 == 0) // Every 30 seconds
		{
			// Update navigation message
			for (i = 0; i < MAX_CHAN; i++)
			{
				if (tp->chan[i].prn > 0)
					generateNavMsg(tp->grx, &(tp->chan[i]), 0,tp->navbit);//将下一帧的数据生成导航电文，主要是更新tow，轨道参数并不会产生变化，在一个Rinex观测周期之后，轨道参数才会发生变化，即ieph++
			}
			navdata_update(look_up);//将导航电文绑定为纹理内存
			// Refresh ephemeris and subframes
			// Quick and dirty fix. Need more elegant way.
			for (sv = 0; sv < MAX_SAT; sv++)
			{
				if (tp->eph[ieph + 1][sv].vflg == 1)//
				{
					dt = subGpsTime(tp->eph[ieph + 1][sv].toc, tp->grx);
					if (dt < SECONDS_IN_HOUR)//小于1小时的时候开始播发下一阶段的导航电文
					{
						ieph++;//

						for (i = 0; i < MAX_CHAN; i++)
						{
							// Generate new subframes if allocated
							if (tp->chan[i].prn != 0)
								eph2sbf(tp->eph[ieph][tp->chan[i].prn - 1], tp->ionoutc, tp->chan[i].sbf);//更新新的一帧 4，5子帧可能会换页
						}
					}

					break;
				}
			}

			// Update channel allocation
			allocateChannel(tp->chan, tp->eph[ieph], tp->ionoutc, tp->grx, tp->xyz[iumd + count_call * 10], elvmask,tp->navbit,tp->allocatedSat);//再次分配通道，30s后可能有的星已经不可见
		}

		// Update receiver time
		tp->grx = incGpsTime(tp->grx, 0.1);

		// Update time counter
		printf("\rRunning time = %4.1f", subGpsTime(tp->grx, tp->g0));

		fflush(stdout);

	}
	//printf("\n");
	//for (int i = 0; i < count * p_n * 10; i++)
	//{
	//	//printf("%f\n", parameters[i]);
	//	printf("%f\n",para_doub[i]);
	//}
	if (count_call == 0)
	{
		CudaStream_firstSec(look_up, parameters, para_doub, dev_noise, sum, samp_freq, count);
	}

	
	
	if (count_call > 0)
	{
		CudaStream(look_up, parameters, para_doub, dev_noise, sum, samp_freq, count);
	}

	//if (count_call == 0)
	//{
	//	//写电文部分
	//	char* str = NULL;
	//	str = (char*)malloc(10);
	//	FILE* feph;
	//	sprintf(str, "i_buff.txt");
	//	feph = fopen(str, "wt+");

	//	//写电文部分
	//	for (int i = 0; i < iq_buff_size * 2 * 9; i++)
	//	{
	//		fprintf(feph, "%f\n ", look_up->i_buff[i]);
	//	}
	//	fclose(feph);
	//}
	
	//产生第一个1s的数据（9个0.1s）
	if (count_call == 0)
	{
		int x = 0;//计数器，统计在一组0.1s数据中，现在是第几个采样点。取值应在0到iq_buff_size-1
		int y = 0;//计数器，统计现在是第几个0.1s的数据
		for (int jj = 0; jj < iq_buff_size * 9; jj++)
		{
			if (x == iq_buff_size)
			{
				x = 0;
				y++;
			}
			buff->real(look_up->i_buff[x + (2 * y) * iq_buff_size]);
			buff->imag(look_up->i_buff[x + (2 * y + 1) * iq_buff_size]);
			buff++;
			x++;
		}
	}


	//产生除第一个1s以外的数据（10个0.1s）
	if (count_call > 0)
	{
		int x = 0;//计数器，统计在一组0.1s数据中，现在是第几个采样点。取值应在0到iq_buff_size-1
		int y = 0;//计数器，统计现在是第几个0.1s的数据
		for (int jj = 0; jj < iq_buff_size * 10; jj++)
		{
			if (x == iq_buff_size)
			{
				x = 0;
				y++;
			}
			buff->real(look_up->i_buff[x + (2 * y) * iq_buff_size]);
			buff->imag(look_up->i_buff[x + (2 * y + 1) * iq_buff_size]);
			buff++;
			x++;
		}
	}



	count_call += 1;

	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	checkCuda(cudaEventElapsedTime(&elapsedTime, starttext, stop));
	printf("\n");
	printf("cuda计时器：程序运行时间：  %3.1f ms\n", elapsedTime);


	GPUMemroy_delete(look_up);
	free(noise);
	checkCuda(cudaFreeHost(dev_noise));
	checkCuda(cudaFreeHost(parameters));
	checkCuda(cudaFreeHost(para_doub));
	checkCuda(cudaFreeHost(sum));
	checkCuda(cudaFree(dev_parameters));
	checkCuda(cudaFree(dev_para_doub));
	checkCuda(cudaFree(dev_sum));
	//checkCuda(cudaFreeHost(dev_buff));
	/*free(parameters);
	free(para_doub);
	free(sum);*/
//	fclose(fpw);
	//free(a1);
//	free(c);
//	free(c1);
	//free(d);
	clock_t tend = clock();

	printf("\nDone!\n");

	// Free I/Q buffer
	delete[]iq_buff;

	free(iq8_buff);

	// Process time
	printf("Process time = %.1f [sec]\n", (double)(tend-tstart)/CLOCKS_PER_SEC);
	//printf("Process time = %.1f [sec]\n", sumTime);
	//return iq_buff_size* loop_count;
	return elapsedTime;
}



