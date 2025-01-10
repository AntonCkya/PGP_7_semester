#include <math.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <set>
#include <stdlib.h>
#include <stdio.h>

#define CSC(call)                                                      \
    {                                                                  \
        cudaError_t status = (call);                                   \
        if (status != cudaSuccess) {                                   \
            const char* msg = cudaGetErrorString(status);              \
            fprintf(stderr, "ERROR by %s:%d! Message: %s\n", __FILE__, \
                    __LINE__, msg);                                    \
            exit(0);                                                   \
        }                                                              \
    }

using namespace std;

const int BLOCKS = 256;
const int THREADS = 256;

const dim3 BLOCKS_2D(64, 64);
const dim3 THREADS_2D(1, 32);

const double EPS = 1e-6;
const double INF = 1e18;

uchar4* read_input(const char* fname, int& w, int& h) {
    FILE* file = fopen(fname, "rb");
    fread(&w, sizeof(int), 1, file);
    fread(&h, sizeof(int), 1, file);
    uchar4* ptr = new uchar4[w * h];
    fread(ptr, sizeof(uchar4), w*h, file);
    fclose(file);
    return ptr;
}

void write_output(const char* fname, int w, int h, uchar4* ptr) {
    FILE* file = fopen(fname, "wb");
    fwrite(&w, sizeof(int), 1, file);
    fwrite(&h, sizeof(int), 1, file);
    fwrite(ptr, sizeof(uchar4), w * h, file);
    fclose(file);
}

template <class T>
struct vector3d {
    T x, y, z;

    __host__ __device__ vector3d(){
        x = 0;
        y = 0;
        z = 0;
    }

    __host__ __device__ vector3d(const T& _x, const T& _y, const T& _z){
        x = _x;
        y = _y;
        z = _z;
    }

    __host__ __device__ vector3d(const vector3d<T>& v){
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __host__ __device__ friend vector3d<T> operator+(const vector3d<T>& a, const vector3d<T>& b) {
        return vector3d(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __host__ __device__ vector3d<T>& operator+=(const vector3d<T>& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ friend vector3d<T> operator-(const vector3d<T>& a, const vector3d<T>& b) {
        return vector3d(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __host__ __device__ vector3d<T>& operator-=(const vector3d<T>& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ friend vector3d<T> operator*(const vector3d<T>& a, const vector3d<T>& b) {
        return vector3d(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    __host__ __device__ vector3d<T>& operator*=(const vector3d<T>& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ friend vector3d<T> operator*(const T& C, const vector3d<T>& v) {
        return vector3d(v.x * C, v.y * C, v.z * C);
    }

    __host__ __device__ vector3d<T>& operator*=(const T& C) {
        x *= C;
        y *= C;
        z *= C;
        return *this;
    }

    __host__ __device__ friend vector3d<T> operator/(const vector3d<T>& v, const T& C) {
        return vector3d(v.x / C, v.y / C, v.z / C);
    }

    __host__ __device__ vector3d<T>& operator/=(const T& C) {
        x /= C;
        y /= C;
        z /= C;
        return *this;
    }

    __host__ __device__ static vector3d<T> from_cyl_coords(const T& r, const T& _z, const T& phi) {
        return vector3d(r * cos(phi), r * sin(phi), _z);
    }

    friend istream& operator>>(istream& in, vector3d<T>& v) {
        in >> v.x >> v.y >> v.z;
        return in;
    }

    __host__ __device__ static double dot(const vector3d<T>& a, const vector3d<T>& b) {
        return (double)(a.x * b.x + a.y * b.y + a.z * b.z);
    }

    __host__ __device__ double len() const {
        return sqrt((double)dot(*this, *this));
    }

    __host__ __device__ void normalize() {
        double l = sqrt(dot(*this, *this));
        x /= l;
        y /= l;
        z /= l;
    }

    __host__ __device__ static vector3d<T> cross(const vector3d<T>& a, const vector3d<T>& b) {
        return vector3d(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
    }

    __host__ __device__ static double square(const vector3d<T>& a, const vector3d<T>& b) {
        return cross(a, b).len();
    }

    __host__ __device__ void crop() {
        if (x > 1.0f) {
            x = 1.0f;
        } else if (x < 0.0f) {
            x = 0.0f;
        }
        if (y > 1.0f) {
            y = 1.0f;
        } else if (y < 0.0f) {
            y = 0.0f;
        }
        if (z > 1.0f) {
            z = 1.0f;
        } else if (z < 0.0f) {
            z = 0.0f;
        }
    }

    __host__ __device__ static vector3d<T> reflect(const vector3d<T>& l, const vector3d<T>& n) {
        vector3d<T> r = l - 2 * dot(n, l) * n;
        r.normalize();
        return r;
    }

    __device__ static void atomicAdd_vec(vector3d<T>* a, const vector3d<T>& b) {
        atomicAdd(&(a->x), b.x);
        atomicAdd(&(a->y), b.y);
        atomicAdd(&(a->z), b.z);
    }
};

using vec3float = vector3d<float>;
using vec3double = vector3d<double>;
using vec3int = vector3d<int>;
using vec3long = vector3d<long>;

struct ttexture {
    int w, h;
    char device;
    uchar4 *data;
    uchar4 *dev_data;

    __host__ __device__ ttexture(){
        w = 0;
        h = 0;
        device = false;
        data = nullptr;
        dev_data = nullptr;
    }

    void load(const char *fname, char _device) {
        device = _device;
        data = read_input(fname, w, h);
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
        CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
    }

    __host__ __device__ ttexture(const ttexture &other){
        w = other.w;
        h = other.h;
        device = other.device;
        data = other.data;
        dev_data = other.dev_data;
    }

    __host__ __device__ vec3float get_pix(double x, double y) const {
        int xp = x * w;
        int yp = y * h;
        xp = max(0, min(xp, w - 1));
        yp = max(0, min(yp, h - 1));
        uchar4 p;
        if (device) {
            p = dev_data[yp * w + xp];
        } else {
            p = data[yp * w + xp];
        }
        vec3float res(p.x, p.y, p.z);
        res /= 255.0f;
        return res;
    }
};

struct ray {
    vec3double p, v;
    int pix_id;
    vec3float coef;

    __host__ __device__ ray(){
        p = vec3double();
        v = vec3double();
        pix_id = 0;
        coef = vec3float();
    }

    __host__ __device__ ray(const vec3double& pp, const vec3double& vv, int ppix_id){
        p = pp;
        v = vv;
        v.normalize();
        pix_id = ppix_id; 
        coef = vec3float(1, 1, 1);
    }

    __host__ __device__ ray(const vec3double& pp, const vec3double& vv, int ppix_id, const vec3float& ccoef){
        p = pp; v = vv;
        v.normalize();
        pix_id = ppix_id; 
        coef = ccoef;
    }
};

struct triangle {
    vec3double a, b, c, n, e1, e2;

    triangle() = delete;

    __host__ __device__ triangle(const vec3double& aa, const vec3double& bb, const vec3double& cc){
        a = aa; b = bb; c = cc;
        n = vec3double::cross(b - a, c - a);
        n.normalize();
        e1 = b - a; e2 = c - a;
    }

    __host__ __device__ void shift(const vec3double& v) {
        a += v; b += v; c += v;
    }

    void check_norm(const vec3double& _n) {
        if (vec3double::dot(_n, n) < -EPS) {
            swap(a, c);
            n = vec3double::cross(b - a, c - a);
            n.normalize();
            e1 = b - a; e2 = c - a;
        }
    }
};

struct polygon {
    triangle trig;
    vec3float color;
    double a, b, c, d;
    float coef_reflection, coef_transparent, coef_blend;
    int n_lights;
    char ttextured;
    vec3double v1, v2, v3;
    ttexture tex;

    polygon() = delete;

    __host__ __device__ polygon(const triangle& triang, const vec3float& ccolor, float ref, float trans)
    : trig(triang) {
        color = ccolor;
        coef_reflection = ref;
        coef_transparent = trans;
        coef_blend = 1.0f - ref - trans;
        n_lights = 0;
        ttextured = false;
        tex = ttexture();
        vec3double p = trig.a;
        vec3double v1 = trig.b - p;
        vec3double v2 = trig.c - p;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -p.x * (v1.y * v2.z - v1.z * v2.y) + (p.y) * (v1.x * v2.z - v1.z * v2.x) + (-p.z) * (v1.x * v2.y - v1.y * v2.x);
    }

    __host__ __device__ polygon(const triangle& triang, const vec3float& ccolor)
    : trig(triang) {
        color = ccolor;
        coef_reflection = 0;
        coef_transparent = 0;
        coef_blend = 1;
        n_lights = 0;
        ttextured = false;
        tex = ttexture();
        vec3double p = trig.a;
        vec3double v1 = trig.b - p;
        vec3double v2 = trig.c - p;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -p.x * (v1.y * v2.z - v1.z * v2.y) + (p.y) * (v1.x * v2.z - v1.z * v2.x) + (-p.z) * (v1.x * v2.y - v1.y * v2.x);
    }

    __host__ __device__ polygon(const triangle& triang, const vec3float& ccolor, int nl, const vec3double& vv1, const vec3double& vv2)
    : trig(triang) {
        color = ccolor;
        coef_reflection = 0;
        coef_transparent = 0;
        coef_blend = 1;
        n_lights = nl;
        ttextured = false;
        v1 = vv1; v2 = vv2;
        tex = ttexture();
        vec3double p = trig.a;
        vec3double v1 = trig.b - p;
        vec3double v2 = trig.c - p;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -p.x * (v1.y * v2.z - v1.z * v2.y) + (p.y) * (v1.x * v2.z - v1.z * v2.x) + (-p.z) * (v1.x * v2.y - v1.y * v2.x);
    }

    __host__ __device__ polygon(const triangle& triang, const vec3float& ccolor, float ref, float trans, const vec3double& vv1,
                            const vec3double& vv2, const vec3double& vv3, const ttexture& textura)
        : trig(triang) {
        color = ccolor;
        coef_reflection = ref;
        coef_transparent = trans;
        coef_blend = 1.0f - ref - trans;
        n_lights = 0;
        ttextured = true;
        v1 = vv1; v2 = vv2; v3 = vv3;
        tex = textura;
        vec3double p = trig.a;
        vec3double v1 = trig.b - p;
        vec3double v2 = trig.c - p;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -p.x * (v1.y * v2.z - v1.z * v2.y) + (p.y) * (v1.x * v2.z - v1.z * v2.x) + (-p.z) * (v1.x * v2.y - v1.y * v2.x);
    }

    __host__ __device__ vec3float get_color(const ray& r, const vec3double& hit) const {
        if (ttextured) {
            vec3double p = hit - v3;
            double beta = (p.x * v1.y - p.y * v1.x) / (v2.x * v1.y - v2.y * v1.x);
            double alpha = (p.x * v2.y - p.y * v2.x) / (v1.x * v2.y - v1.y * v2.x);
            return tex.get_pix(alpha, beta);
        } else if (n_lights > 0 && vec3double::dot(trig.n, r.v) > 0.0) {
            vec3double vl = (v2 - v1) / (n_lights + 1);
            for (int i = 1; i <= n_lights; ++i) {
                vec3double p_light = v1 + i * vl;
                if ((p_light - hit).len() < 0.025) {
                    return vec3float(16.0f, 16.0f, 16.0f);
                }
            }
        }
        return color;
    }
};

__host__ __device__ void intersect_ray_and_plane(const ray& r, const polygon& poly, double& _t) {
    _t = (poly.a * r.p.x + poly.b * r.p.y + poly.c * r.p.z + poly.d) / (poly.a * r.v.x + poly.b * r.v.y + poly.c * r.v.z);
    _t *= -1;
}

__host__ __device__ void intersect_ray_and_polygon(const ray& r, const polygon& poly, double& tt, bool& solve) {
    vec3double PP = vec3double::cross(r.v, poly.trig.e2);
    double div = vec3double::dot(PP, poly.trig.e1);
    if (fabs(div) < EPS) {
        solve = false;
        return;
    }
    vec3double TT = r.p - poly.trig.a;
    double u = vec3double::dot(PP, TT) / div;
    if (u < 0.0 || u > 1.0) {
        solve = false;
        return;
    }
    vec3double QQ = vec3double::cross(TT, poly.trig.e1);
    double v = vec3double::dot(QQ, r.v) / div;
    if (v < 0.0 || u + v > 1.0) {
        solve = false;
        return;
    }
    tt = vec3double::dot(QQ, poly.trig.e2) / div;
    if (tt < 0.0){
        solve = false;
        return;
    }
    solve = true;
}

struct figure {
    vec3double center;
    vec3float color;
    double radius;
    double coef_reflection, coef_transparent, n_lights;
    friend istream& operator>>(istream& in, figure& f) {
        in >> f.center >> f.color >> f.radius >> f.coef_reflection >>
            f.coef_transparent >> f.n_lights;
        return in;
    }
};

struct light_source {
    vec3double p;
    vec3float i;
    friend istream& operator>>(istream& in, light_source& ls) {
        in >> ls.p >> ls.i;
        return in;
    }
};

char device_gpu;
int n_polys;
vector<polygon> polys;
int frames;
string path;
int w, h;
double angle;
double r0_c, z0_c, phi0_c, Ar_c, Az_c, wr_c, wz_c, wphi_c, pr_c, pz_c;
double r0_n, z0_n, phi0_n, Ar_n, Az_n, wr_n, wz_n, wphi_n, pr_n, pz_n;
vector<figure> figs(3);
vector<vec3double> floor_p(4);
string ttexture_floor;
vec3float floor_color;
double floor_refl;
int n_sources;
vector<light_source> lights;

void read_obj(int id, string fname) {
    const vec3float EDGE_COLOR(0.05f, 0.05f, 0.05f);
    ifstream f(fname);
    vector<vec3double> verts;
    vector<set<int>> verts_polygons;
    vector<vec3int> polygons;
    int poly_id = 0;
    string s;
    while (f >> s) {
        if (s == "v") {
            vec3double vertex;
            f >> vertex;
            vertex *= figs[id].radius;
            verts.push_back(vertex);
            verts_polygons.push_back(set<int>());
        } else if (s == "f") {
            vec3int ids;
            f >> ids;
            --ids.x; --ids.y; --ids.z;
            polygons.push_back(ids);
            verts_polygons[ids.x].insert(poly_id);
            verts_polygons[ids.y].insert(poly_id);
            verts_polygons[ids.z].insert(poly_id);
            ++poly_id;
        }
    }
    f.close();
    double side = INF;
    int m = verts.size();
    for (int i = 0; i < m; ++i) {
        vec3double vi = verts[i];
        for (int j = i + 1; j < m; ++j) {
            vec3double vj = verts[j];
            side = min(side, (vi - vj).len());
        }
    }
    set<int> unique_poly_ids;
    for (int i = 0; i < m; ++i) {
        vec3double vi = verts[i];
        for (int j = i + 1; j < m; ++j) {
            vec3double vj = verts[j];
            if ((vi - vj).len() > side + EPS) {
                continue;
            }
            vector<int> trig_ids;
            vector<triangle> trigs;
            for (int elem : verts_polygons[i]) {
                if (verts_polygons[j].count(elem)) {
                    trig_ids.push_back(elem);
                    vec3int ids = polygons[elem];
                    trigs.push_back(triangle(verts[ids.x], verts[ids.y], verts[ids.z]));
                }
            }
            double t;
            int id1 = trig_ids[0]; int id2 = trig_ids[1];
            triangle trig1 = trigs[0]; triangle trig2 = trigs[1];
            vec3double n1 = 0.05 * trig1.n; vec3double n2 = 0.05 * trig2.n;
            vec3double n_avg = (n1 + n2) / 2;
            trig1.shift(n1); trig2.shift(n2);
            vec3double vi1 = vi + n1; vec3double vi2 = vi + n2;
            vec3double vj1 = vj + n1; vec3double vj2 = vj + n2;
            vec3double vi_avg = (vi1 + vi2) / 2 + figs[id].center;
            vec3double vj_avg = (vj1 + vj2) / 2 + figs[id].center;
            triangle edge1(vi1, vj2, vi2);
            edge1.check_norm(n_avg);
            intersect_ray_and_plane(ray(vec3double(0, 0, 0), vi, -1), polygon(edge1, EDGE_COLOR), t);
            triangle corneri(vi1, vi2, t * vi / vi.len());
            corneri.check_norm(n_avg);
            edge1.shift(figs[id].center);
            corneri.shift(figs[id].center);
            polys.push_back(polygon(edge1, EDGE_COLOR, figs[id].n_lights, vi_avg, vj_avg));
            polys.push_back(polygon(corneri, EDGE_COLOR));
            triangle edge2(vi1, vj1, vj2);
            edge2.check_norm(n_avg);
            intersect_ray_and_plane(ray(vec3double(0, 0, 0), vj, -1), polygon(edge2, EDGE_COLOR), t);
            triangle cornerj(vj1, t * vj / vj.len(), vj2);
            cornerj.check_norm(n_avg);
            edge2.shift(figs[id].center);
            cornerj.shift(figs[id].center);
            polys.push_back(polygon(edge2, EDGE_COLOR, figs[id].n_lights, vi_avg, vj_avg));
            polys.push_back(polygon(cornerj, EDGE_COLOR));

            if (!unique_poly_ids.count(id1)) {
                trig1.shift(figs[id].center);
                polys.push_back(polygon(trig1, figs[id].color, figs[id].coef_reflection, figs[id].coef_transparent));
                unique_poly_ids.insert(id1);
            }
            if (!unique_poly_ids.count(id2)) {
                trig2.shift(figs[id].center);
                polys.push_back(polygon(trig2, figs[id].color, figs[id].coef_reflection, figs[id].coef_transparent));
                unique_poly_ids.insert(id2);
            }
        }
    }
}

ttexture t_ttexture_floor;

void scene() { 
    t_ttexture_floor.load(ttexture_floor.c_str(), device_gpu);
    triangle t1 = {floor_p[0], floor_p[2], floor_p[1]};
    triangle t2 = {floor_p[2], floor_p[0], floor_p[3]};
    polys.push_back(polygon(t1, floor_color, floor_refl, 0.0, floor_p[1] - floor_p[2], floor_p[1] - floor_p[0],
                        floor_p[0] + floor_p[2] - floor_p[1], t_ttexture_floor));
    polys.push_back(polygon(t2, floor_color, floor_refl, 0.0, floor_p[0] - floor_p[3], floor_p[2] - floor_p[3],
                        floor_p[3], t_ttexture_floor));
    read_obj(0, "icosahedron.obj");
    read_obj(1, "tetrahedron.obj");
    read_obj(2, "dodecahedron.obj");
    n_polys = polys.size();
}

__host__ __device__ vec3double mult(const vec3double& a, const vec3double& b, const vec3double& c, const vec3double& v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

__host__ __device__ vec3float shading(const ray r, const vec3double hit, const polygon poly, int id,
                const light_source* lights, int n_sources, const polygon* polys, int n_polys) {
    vec3float polygon_color = poly.get_color(r, hit);
    vec3float clr = 0.25 * poly.coef_blend * r.coef * polygon_color;
    for (int j = 0; j < n_sources; ++j) {
        ray r_light(hit, lights[j].p - hit, r.pix_id);
        vec3float coef_vis(1.0f, 1.0f, 1.0f);
        double t_max = (lights[j].p - hit).len();
        for (int i = 0; i < n_polys; ++i) {
            double t;
            bool flag;
            if (i == id) {
                continue;
            }
            intersect_ray_and_polygon(r_light, polys[i], t, flag);
            if (flag and t < t_max) {
                coef_vis *= polys[i].coef_transparent;
            }
        }
        vec3float a_clr = poly.coef_blend * lights[j].i * r.coef * coef_vis * polygon_color;
        double coef_diff = vec3double::dot(poly.trig.n, r_light.v);
        double coef_spec = 0.0;
        if (coef_diff < 0.0) {
            coef_diff = 0.0;
        } else {
            vec3double reflected = vec3double::reflect(r_light.v, poly.trig.n);
            coef_spec = vec3double::dot(reflected, r.v);
            if (coef_spec < 0.0) {
                coef_spec = 0.0;
            } else {
                coef_spec = coef_spec*coef_spec*coef_spec*coef_spec*coef_spec*coef_spec*coef_spec*coef_spec*coef_spec;
            }
        }
        clr += (0.5 * coef_spec + coef_diff) * a_clr;
    }
    clr.crop();
    return clr;
}

void clear_data_cpu(vec3float* data, const int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = vec3float(0, 0, 0);
    }
}

void trace_cpu(const ray* in_rays, int size_in, ray* out_rays, int& size_out, vec3float* data_cpu) {
    for (int k = 0; k < size_in; ++k) {
        int i_min = n_polys;
        double t_min = INF;
        for (int i = 0; i < n_polys; i++) {
            double t;
            bool flag;
            intersect_ray_and_polygon(in_rays[k], polys[i], t, flag);
            if (flag && t < t_min) {
                i_min = i;
                t_min = t;
            }
        }
        if (i_min == n_polys) {
            continue;
        }
        vec3double hit = t_min * in_rays[k].v + in_rays[k].p;
        data_cpu[in_rays[k].pix_id] += shading(in_rays[k], hit, polys[i_min], i_min, lights.data(), n_sources, polys.data(), n_polys);
        if (polys[i_min].coef_transparent > 0) {
            out_rays[size_out++] = ray(hit + 1e-3 * in_rays[k].v, in_rays[k].v, in_rays[k].pix_id,
                    polys[i_min].coef_transparent * in_rays[k].coef * polys[i_min].get_color(in_rays[k], hit));
        }
        if (polys[i_min].coef_reflection > 0) {
            vec3double reflected = vec3double::reflect(in_rays[k].v, polys[i_min].trig.n);
            out_rays[size_out++] = ray(hit + 1e-3 * reflected, reflected, in_rays[k].pix_id,
                    polys[i_min].coef_reflection * in_rays[k].coef * polys[i_min].get_color(in_rays[k], hit));
        }
    }
}

void init_rays_cpu(const vec3double& c, const vec3double& v, int w, int h, double angle, ray* rays) {
    vec3double dz = v - c;
    vec3double dx = vec3double::cross(dz, vec3double(0, 0, 1));
    vec3double dy = vec3double::cross(dx, dz);
    dx.normalize(); dy.normalize(); dz.normalize();
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3double v(-1.0 + (2.0 / (w - 1.0)) * i, (-1.0 + (2.0 / (h - 1.0)) * j) * h / w, 1.0 / tan(angle * M_PI / 360.0));
            vec3double dir = mult(dx, dy, dz, v);
            int pix_id = i + (h - 1 - j) * w;
            rays[j + i * h] = ray(c, dir, pix_id);
        }
    }
}

void write_data_cpu(uchar4* data, vec3float* data_vec3float, int size) {
    for (int i = 0; i < size; ++i) {
        data_vec3float[i].crop();
        data_vec3float[i] *= 255.0f;
        data[i] = make_uchar4(data_vec3float[i].x, data_vec3float[i].y, data_vec3float[i].z, 255);
    }
}

int rec_depth, coef_ssaa;

void render_cpu(int frame_id, const vec3double& c, const vec3double& v, int w, int h, double angle, uchar4* data) {
    int size_in = w * h;
    long long total_rays = 0;
    vec3float* data_vec3float = new vec3float[size_in];
    clear_data_cpu(data_vec3float, w * h);
    ray* ray_in = new ray[size_in];
    init_rays_cpu(c, v, w, h, angle, ray_in);
    cudaEvent_t tstart, tstop;
    CSC(cudaEventCreate(&tstart));
    CSC(cudaEventCreate(&tstop));
    CSC(cudaEventRecord(tstart));
    for (int rec = 0; rec < rec_depth and size_in; ++rec) {
        total_rays += size_in;
        ray* ray_out = new ray[2 * size_in];
        int size_out = 0;
        trace_cpu(ray_in, size_in, ray_out, size_out, data_vec3float);
        delete[] ray_in;
        ray_in = ray_out;
        size_in = size_out;
    }
    write_data_cpu(data, data_vec3float, w * h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(tstop));
    CSC(cudaEventSynchronize(tstop));
    float ms;
    CSC(cudaEventElapsedTime(&ms, tstart, tstop));
    delete[] ray_in;
    delete[] data_vec3float;
    printf("%d\t%.3lf\t%lli\n", frame_id, ms, total_rays);
    CSC(cudaEventDestroy(tstart));
    CSC(cudaEventDestroy(tstop));
}

__global__ void clear_data_gpu(vec3float* dev_data, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += offset) {
        dev_data[i] = vec3float(0, 0, 0);
    }
}

light_source* dev_lights;
polygon* dev_polygons;

__global__ void trace_gpu(const ray* in_rays, const int size_in, ray* out_rays, int* size_out, vec3float* dev_data,
                        const light_source* dev_lights, int n_sources, const polygon* dev_polygons, int n_polys) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int k = idx; k < size_in; k += offset) {
        int i_min = n_polys;
        double t_min = INF;
        for (int i = 0; i < n_polys; i++) {
            double t;
            bool flag;
            intersect_ray_and_polygon(in_rays[k], dev_polygons[i], t, flag);
            if (flag && t < t_min) {
                i_min = i;
                t_min = t;
            }
        }
        if (i_min == n_polys) {
            continue;
        }
        vec3double hit = t_min * in_rays[k].v + in_rays[k].p;
        vec3float polygon_color = dev_polygons[i_min].get_color(in_rays[k], hit);
        vec3float::atomicAdd_vec(&dev_data[in_rays[k].pix_id], shading(in_rays[k], hit, dev_polygons[i_min], i_min, dev_lights, n_sources, dev_polygons, n_polys));
        if (dev_polygons[i_min].coef_transparent > 0) {
            out_rays[atomicAdd(size_out, 1)] = ray(hit + 1e-3 * in_rays[k].v, in_rays[k].v, in_rays[k].pix_id,
                dev_polygons[i_min].coef_transparent * in_rays[k].coef * polygon_color);
        }
        if (dev_polygons[i_min].coef_reflection > 0) {
            vec3double reflected = vec3double::reflect(in_rays[k].v, dev_polygons[i_min].trig.n);
            out_rays[atomicAdd(size_out, 1)] = ray(hit + 1e-3 * reflected, reflected, in_rays[k].pix_id,
                dev_polygons[i_min].coef_reflection * in_rays[k].coef * polygon_color);
        }
    }
}

__global__ void init_rays_gpu(const vec3double c, const vec3double v, int w, int h, double angle, ray* dev_rays) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;
    vec3double dz = v - c;
    vec3double dx = vec3double::cross(dz, vec3double(0, 0, 1));
    vec3double dy = vec3double::cross(dx, dz);
    dx.normalize(); dy.normalize(); dz.normalize();
    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3double v(-1.0 + (2.0 / (w - 1.0)) * i, (-1.0 + (2.0 / (h - 1.0)) * j) * h / w, 1.0 / tan(angle * M_PI / 360.0));
            vec3double dir = mult(dx, dy, dz, v);
            int pix_id = i + (h - 1 - j) * w;
            dev_rays[j + i * h] = ray(c, dir, pix_id);
        }
    }
}

__global__ void write_data_gpu(uchar4* dev_data, vec3float* dev_data_vec3float, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += offset) {
        dev_data_vec3float[i].crop();
        dev_data_vec3float[i] *= 255.0f;
        dev_data[i] = make_uchar4(dev_data_vec3float[i].x, dev_data_vec3float[i].y, dev_data_vec3float[i].z, 255);
    }
}

void render_gpu(int frame_id, const vec3double c, const vec3double v, int w, int h, double angle, uchar4* dev_data) {
    int size_in = w * h;
    const int zeroptr = 0;
    long long total_rays = 0;
    vec3float* dev_data_vec3float;
    CSC(cudaMalloc(&dev_data_vec3float, sizeof(vec3float) * size_in));
    clear_data_gpu<<<BLOCKS, THREADS>>>(dev_data_vec3float, w * h);
    ray* dev_ray_in;
    CSC(cudaMalloc(&dev_ray_in, sizeof(ray) * size_in));
    init_rays_gpu<<<BLOCKS_2D, THREADS_2D>>>(c, v, w, h, angle, dev_ray_in);
    CSC(cudaGetLastError());
    cudaEvent_t tstart, tstop;
    CSC(cudaEventCreate(&tstart));
    CSC(cudaEventCreate(&tstop));
    CSC(cudaEventRecord(tstart));
    for (int rec = 0; rec < rec_depth and size_in; ++rec) {
        total_rays += size_in;
        ray* dev_ray_out;
        CSC(cudaMalloc(&dev_ray_out, 2 * sizeof(ray) * size_in));
        int* size_out;
        CSC(cudaMalloc(&size_out, sizeof(int)));
        CSC(cudaMemcpy(size_out, &zeroptr, sizeof(int), cudaMemcpyHostToDevice));
        trace_gpu<<<BLOCKS, THREADS>>>(dev_ray_in, size_in, dev_ray_out, size_out, dev_data_vec3float, dev_lights, n_sources,
                                    dev_polygons, n_polys);
        CSC(cudaGetLastError());
        CSC(cudaFree(dev_ray_in));
        dev_ray_in = dev_ray_out;
        CSC(cudaMemcpy(&size_in, size_out, sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaFree(size_out));
        CSC(cudaGetLastError());
    }
    write_data_gpu<<<BLOCKS, THREADS>>>(dev_data, dev_data_vec3float, w * h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(tstop));
    CSC(cudaEventSynchronize(tstop));
    float ms;
    CSC(cudaEventElapsedTime(&ms, tstart, tstop));
    CSC(cudaFree(dev_ray_in));
    CSC(cudaFree(dev_data_vec3float));
    CSC(cudaGetLastError());
    printf("%d\t%.3lf\t%lli\n", frame_id, ms, total_rays);
    CSC(cudaEventDestroy(tstart));
    CSC(cudaEventDestroy(tstop));
    fflush(stdout);
}

void ssaa_cpu(const uchar4 *input, uchar4 *output, int w, int h, int c) {
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            double r = 0.0, g = 0.0, b = 0.0;
            for (int k = 0; k < c; ++k) {
                for (int l = 0; l < c; ++l) {
                    uchar4 p = input[(c * j + l)*(w * c) + c * i + k];
                    r += p.x;
                    g += p.y;
                    b += p.z;
                }
            }
            r /= c * c;
            g /= c * c;
            b /= c * c;
            output[i + j * w] = make_uchar4(r, g, b, 255);
        }
    }
}

__global__ void ssaa_gpu(const uchar4 *input, uchar4 *output, int w, int h, int c) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            double r = 0.0, g = 0.0, b = 0.0;
            for (int k = 0; k < c; ++k) {
                for (int l = 0; l < c; ++l) {
                    uchar4 p = input[(c * j + l)*(w * c) + c * i + k];
                    r += p.x;
                    g += p.y;
                    b += p.z;
                }
            }
            r /= c * c;
            g /= c * c;
            b /= c * c;
            output[i + j * w] = make_uchar4(r, g, b, 255);
        }
    }
}

void read_input_params() {
    cin >> frames;
    cin >> path;
    cin >> w >> h >> angle;
    cin >> r0_c >> z0_c >> phi0_c >> Ar_c >> Az_c >> wr_c >> wz_c >> wphi_c >> pr_c >> pz_c;
    cin >> r0_n >> z0_n >> phi0_n >> Ar_n >> Az_n >> wr_n >> wz_n >> wphi_n >> pr_n >> pz_n;
    for (int i = 0; i < 3; ++i) {
        cin >> figs[i];
    }
    for (int i = 0; i < 4; ++i) {
        cin >> floor_p[i];
    }
    cin >> ttexture_floor >> floor_color >> floor_refl;
    cin >> n_sources;
    lights.resize(n_sources);
    for (int i = 0; i < n_sources; ++i) {
        cin >> lights[i];
    }
    cin >> rec_depth >> coef_ssaa;
}

void print_file(const char *fname) {
    FILE *in = fopen(fname, "r");
    while (!feof(in)) {
        char c = getc(in);
        if (c == EOF) {
            break;
        }
        printf("%c", c);
    }
    fclose(in);
}

int main(int argc, char *argv[]) {
    device_gpu = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) {
            device_gpu = 0;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            device_gpu = 1;
        } else if (strcmp(argv[i], "--default") == 0) {
            print_file("default.in");
            return 0;
        } else {
            printf("Unknown key: %s\n", argv[i]);
            return 0;
        }
    }
    read_input_params();
    scene();
    CSC(cudaMalloc(&dev_lights, sizeof(light_source) * n_sources));
    CSC(cudaMemcpy(dev_lights, lights.data(), sizeof(light_source) * n_sources, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_polygons, sizeof(polygon) * n_polys));
    CSC(cudaMemcpy(dev_polygons, polys.data(), sizeof(polygon) * n_polys, cudaMemcpyHostToDevice));
    int w_ssaa = w * coef_ssaa;
    int h_ssaa = h * coef_ssaa;
    char buffer[256];
    uchar4 *data_ssaa = new uchar4[w_ssaa * h_ssaa];
    uchar4 *data = new uchar4[w * h];
    uchar4 *dev_data_ssaa;
    CSC(cudaMalloc(&dev_data_ssaa, sizeof(uchar4) * w_ssaa * h_ssaa));
    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    for (int k = 0; k < frames; k++) {
        vec3double c, v;
        c = vec3double::from_cyl_coords(r0_c + Ar_c * sin(wr_c * k * (2 * M_PI / frames) + pr_c), z0_c + Az_c * sin(wz_c * k * (2 * M_PI / frames) + pz_c),
            phi0_c + wphi_c * k * (2 * M_PI / frames));
        v = vec3double::from_cyl_coords(r0_n + Ar_n * sin(wr_n * k * (2 * M_PI / frames) + pr_n), z0_n + Az_n * sin(wz_n * k * (2 * M_PI / frames) + pz_n),
            phi0_n + wphi_n * k * (2 * M_PI / frames));
        if (device_gpu > 0) {
            render_gpu(k, c, v, w_ssaa, h_ssaa, angle, dev_data_ssaa);
            ssaa_gpu<<<BLOCKS_2D, THREADS_2D>>>(dev_data_ssaa, dev_data, w, h, coef_ssaa);
            CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } else {
            render_cpu(k, c, v, w_ssaa, h_ssaa, angle, data_ssaa);
            ssaa_cpu(data_ssaa, data, w, h, coef_ssaa);
        }
        sprintf(buffer, path.c_str(), k);
        write_output(buffer, w, h, data);
    }
    CSC(cudaFree(dev_data_ssaa));
    CSC(cudaFree(dev_data));
    free(data);
    CSC(cudaFree(dev_lights));
    CSC(cudaFree(dev_polygons));
    CSC(cudaGetLastError());
    return 0;
}
