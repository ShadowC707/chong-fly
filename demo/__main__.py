import argparse


def main():
    parser=argparse.ArgumentParser(description='Chong-Fly local simulation and recorded replay')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--record-examples',action='store_true')
    parser.add_argument('--verify-examples',action='store_true')
    parser.add_argument('--verify-recording',type=str)
    parser.add_argument('--container-sitl',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.verify_recording:
        from .recording import load_recording
        data=load_recording(args.verify_recording)
        if data['manifest'].get('backend') == 'sitl':
            from .sitl import recompute
        else:
            from .examples import recompute
        rebuilt=recompute(data)
        if rebuilt['frames']!=data['frames']:
            raise SystemExit('Numerical trace differs from recorded frames')
        print('EXACT numerical trace match:',args.verify_recording)
    elif args.record_examples or args.verify_examples:
        from .examples import examples
        examples(verify=args.verify_examples)
    else:
        from .server import serve
        serve(args.port, container_sitl=args.container_sitl)


if __name__=='__main__': main()
